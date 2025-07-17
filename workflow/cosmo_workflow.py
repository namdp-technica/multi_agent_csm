import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Sequence

from google.adk.agents import LlmAgent, BaseAgent
from google.genai import types
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
import asyncio

logger = logging.getLogger(__name__)

class CosmoWorkflow(BaseAgent):
    """
    Custom orchestrator for Cosmo workflow that breaks down tasks using a main agent
    and executes them in parallel using sub-agents for vector database search.
    """
    
    # Pydantic model configuration to allow arbitrary attributes
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def __init__(self, name: str, main_agent: LlmAgent, sub_agents_list: list[LlmAgent]):
        """
        Initialize CosmoWorkflow with main agent and sub-agents.
        
        Args:
            name: Name of the workflow
            main_agent: Main agent for task decomposition and evaluation
            sub_agents_list: List of sub-agents for parallel execution
        """
        # Convert to BaseAgent list to satisfy type checker
        all_agents: Sequence[BaseAgent] = [main_agent] + sub_agents_list
        
        # Call super().__init__ with only the parameters BaseAgent accepts
        super().__init__(
            name=name,
            sub_agents=list(all_agents),
        )
        
        # Store agents as instance attributes
        self.main_agent = main_agent
        self.sub_agents_list = sub_agents_list

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Main workflow implementation:
        1. Main agent breaks down user input into tasks
        2. Sub-agents execute tasks in parallel
        3. Main agent evaluates results
        4. Retry if needed
        5. Provide summary and follow-up question
        """

        # Record workflow start time
        workflow_start_time = time.time()
        workflow_start_datetime = datetime.now()
        
        logger.info(f"[{self.name}] Starting Cosmo workflow.")
        logger.info(f"[{self.name}] ⏰ Workflow start time: {workflow_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Task decomposition by main agent
        step1_start = time.time()
        logger.info(f"[{self.name}] 📋 Step 1: Running main agent for task decomposition...")
        async for event in self.main_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from MainAgent: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        
        step1_duration = time.time() - step1_start
        logger.info(f"[{self.name}] ✅ Step 1 completed in {step1_duration:.2f} seconds")

        # Check if task list was generated (it might be in evaluation_result for first run)
        task_list_data = None
        if "task_list" in ctx.session.state and ctx.session.state["task_list"]:
            task_list_data = ctx.session.state["task_list"]
        elif "evaluation_result" in ctx.session.state and ctx.session.state["evaluation_result"]:
            task_list_data = ctx.session.state["evaluation_result"]
        
        if not task_list_data:
            logger.error(f"[{self.name}] No task list generated. Aborting workflow.")
            return
            
        logger.info(f"[{self.name}] Task list data: {task_list_data}")
        
        # Parse the task list from JSON string to list
        import json
        try:
            if isinstance(task_list_data, str):
                task_list = json.loads(task_list_data)
            else:
                task_list = task_list_data
                
            # If it's an evaluation result, extract the task list
            if isinstance(task_list, dict) and "actions" in task_list:
                logger.info(f"[{self.name}] This is an evaluation result, not initial task list. Skipping sub-agent execution.")
                task_list = []
            elif not isinstance(task_list, list):
                logger.error(f"[{self.name}] Expected task list to be a list, got: {type(task_list)}")
                return
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[{self.name}] Failed to parse task list: {e}")
            return

        # 2. Execute sub-agents in parallel
        step2_start = time.time()
        logger.info(f"[{self.name}] 🔄 Step 2: Running sub-agents in parallel...")
        
        async def run_sub_agent_task(sub_agent: LlmAgent, query: str):
            """Run a single sub-agent with a specific query"""
            # Store the query in session state for the sub-agent to use
            original_query = ctx.session.state.get("current_query", "")
            ctx.session.state["current_query"] = query
            
            # Run the sub-agent with the context
            async for event in sub_agent.run_async(ctx):
                logger.info(f"[{self.name}] Event from {sub_agent.name}: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            
            # Restore original query
            ctx.session.state["current_query"] = original_query

        # Create parallel tasks for sub-agents
        parallel_tasks = []
        
        # Debug: Check task_list type and content
        logger.info(f"[{self.name}] Task list type: {type(task_list)}")
        logger.info(f"[{self.name}] Task list content: {task_list}")
        
        # Ensure task_list is a list
        if not isinstance(task_list, list):
            logger.error(f"[{self.name}] Expected task_list to be a list, got {type(task_list)}")
            return
            
        for i, task in enumerate(task_list):
            logger.info(f"[{self.name}] Processing task {i}: {task} (type: {type(task)})")
            
            # Ensure task is a dictionary
            if not isinstance(task, dict):
                logger.error(f"[{self.name}] Expected task to be a dict, got {type(task)}: {task}")
                continue
                
            # Check if required keys exist
            if "agent" not in task or "query" not in task:
                logger.error(f"[{self.name}] Task missing required keys: {task}")
                continue
            
            # Find the appropriate sub-agent for this task
            sub_agent = next((agent for agent in self.sub_agents_list if agent.name == task["agent"]), None)
            if sub_agent:
                parallel_tasks.append(run_sub_agent_task(sub_agent, task["query"]))
            else:
                logger.warning(f"[{self.name}] Sub-agent not found: {task['agent']}")

        # Execute parallel tasks
        if parallel_tasks:
            async for event in self._merge_parallel_events(parallel_tasks):
                yield event
        
        step2_duration = time.time() - step2_start
        logger.info(f"[{self.name}] ✅ Step 2 completed in {step2_duration:.2f} seconds")

        # 3. Main agent evaluates results
        step3_start = time.time()
        logger.info(f"[{self.name}] 🔍 Step 3: Sending results to main agent for evaluation...")
        async for event in self.main_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from MainAgent (Evaluation): {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        
        step3_duration = time.time() - step3_start
        logger.info(f"[{self.name}] ✅ Step 3 completed in {step3_duration:.2f} seconds")

        # Check if evaluation was successful
        if "evaluation_result" not in ctx.session.state:
            logger.error(f"[{self.name}] Evaluation result missing. Aborting workflow.")
            return

        # Parse evaluation result from JSON string to dict
        evaluation_str = ctx.session.state["evaluation_result"]
        try:
            if isinstance(evaluation_str, str):
                # Remove markdown code blocks if present
                evaluation_str = evaluation_str.strip()
                if evaluation_str.startswith("```json"):
                    evaluation_str = evaluation_str[7:]
                if evaluation_str.endswith("```"):
                    evaluation_str = evaluation_str[:-3]
                evaluation = json.loads(evaluation_str.strip())
            else:
                evaluation = evaluation_str
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[{self.name}] Failed to parse evaluation result: {e}")
            logger.error(f"[{self.name}] Raw evaluation result: {evaluation_str}")
            return

        # 4. Retry mechanism if needed
        step4_start = time.time()
        retry_actions = evaluation.get("actions", [])
        if retry_actions:
            logger.info(f"[{self.name}] 🔄 Step 4: Running retry sub-agents in parallel...")
            retry_tasks = []
            
            for retry in retry_actions:
                sub_agent = next((agent for agent in self.sub_agents_list if agent.name == retry["agent"]), None)
                if sub_agent:
                    retry_tasks.append(run_sub_agent_task(sub_agent, retry["new_query"]))
                else:
                    logger.warning(f"[{self.name}] Retry sub-agent not found: {retry['agent']}")

            if retry_tasks:
                async for event in self._merge_parallel_events(retry_tasks):
                    yield event
                
                # After retry, run main agent again for final evaluation
                logger.info(f"[{self.name}] 🔍 Running final evaluation after retry...")
                async for event in self.main_agent.run_async(ctx):
                    logger.info(f"[{self.name}] Event from MainAgent (Final): {event.model_dump_json(indent=2, exclude_none=True)}")
                    yield event
        
        step4_duration = time.time() - step4_start
        if retry_actions:
            logger.info(f"[{self.name}] ✅ Step 4 completed in {step4_duration:.2f} seconds")
        else:
            logger.info(f"[{self.name}] ⏭️ Step 4 skipped (no retry needed)")

        # 5. Final summary and follow-up question
        final_evaluation = ctx.session.state.get("evaluation_result", evaluation)
        if isinstance(final_evaluation, str):
            try:
                final_evaluation_str = final_evaluation.strip()
                if final_evaluation_str.startswith("```json"):
                    final_evaluation_str = final_evaluation_str[7:]
                if final_evaluation_str.endswith("```"):
                    final_evaluation_str = final_evaluation_str[:-3]
                final_evaluation = json.loads(final_evaluation_str.strip())
            except:
                final_evaluation = evaluation
        
        summary = final_evaluation.get("summary", "Workflow completed successfully.")
        followup_question = final_evaluation.get("followup_question", "Do you have any other questions?")
        
        yield Event(
            author=self.name,
            content=types.Content(role="assistant", parts=[
                types.Part(text=f"{summary}\n\n🤔 {followup_question}")
            ])
        )

        # Calculate total workflow time
        workflow_end_time = time.time()
        workflow_end_datetime = datetime.now()
        total_workflow_time = workflow_end_time - workflow_start_time
        
        logger.info(f"[{self.name}] 🏁 Workflow finished.")
        logger.info(f"[{self.name}] ⏰ Workflow end time: {workflow_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"[{self.name}] ⏱️ Total workflow time: {total_workflow_time:.2f} seconds")
        
        # Store timing info in session state
        ctx.session.state["workflow_timing"] = {
            "start_time": workflow_start_datetime.isoformat(),
            "end_time": workflow_end_datetime.isoformat(),
            "total_time": total_workflow_time,
            "step1_duration": step1_duration,
            "step2_duration": step2_duration,
            "step3_duration": step3_duration,
            "step4_duration": step4_duration if retry_actions else 0
        }

    async def _merge_parallel_events(self, task_generators: list[AsyncGenerator[Event, None]]) -> AsyncGenerator[Event, None]:
        """
        Merge events from multiple parallel async generators.
        This allows sub-agents to run in parallel while yielding events as they complete.
        """
        if not task_generators:
            return
            
        # Create tasks for each generator
        tasks = []
        for i, gen in enumerate(task_generators):
            task = asyncio.create_task(gen.__anext__())
            tasks.append((task, i, gen))
        
        # Process completed tasks
        while tasks:
            # Wait for at least one task to complete
            completed_tasks = []
            pending_tasks = []
            
            for task, idx, gen in tasks:
                if task.done():
                    completed_tasks.append((task, idx, gen))
                else:
                    pending_tasks.append((task, idx, gen))
            
            # If no tasks are done, wait for the first one
            if not completed_tasks:
                done, pending = await asyncio.wait([task for task, _, _ in tasks], return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    # Find the corresponding generator
                    for t, idx, gen in tasks:
                        if t == task:
                            completed_tasks.append((t, idx, gen))
                            break
                
                # Update pending tasks
                pending_tasks = [(t, idx, gen) for t, idx, gen in tasks if t in pending]
            
            # Process completed tasks
            new_tasks = []
            for task, idx, gen in completed_tasks:
                try:
                    event = task.result()
                    yield event
                    # Create new task for this generator
                    new_task = asyncio.create_task(gen.__anext__())
                    new_tasks.append((new_task, idx, gen))
                except StopAsyncIteration:
                    # This generator is exhausted
                    continue
            
            # Update tasks list
            tasks = pending_tasks + new_tasks
