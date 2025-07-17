import json
from typing import AsyncGenerator, Optional
from pydantic import PrivateAttr
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext


from typing import override
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_branch_ctx_for_sub_agent(
    agent: BaseAgent,
    sub_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
    """Creates an isolated context per sub-agent."""
    invocation_context = invocation_context.model_copy()
    branch_suffix = f"{agent.name}.{sub_agent.name}"
    invocation_context.branch = (
        f"{invocation_context.branch}.{branch_suffix}"
        if invocation_context.branch
        else branch_suffix
    )
    return invocation_context

async def _merge_agent_run(agent_runs: list[AsyncGenerator[Event, None]]) -> AsyncGenerator[Event, None]:
    """Merges all sub-agent runs."""
    import asyncio
    tasks = [asyncio.create_task(run.__anext__()) for run in agent_runs]
    pending_tasks = set(tasks)

    while pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                yield task.result()
                for i, original in enumerate(tasks):
                    if task == original:
                        new_task = asyncio.create_task(agent_runs[i].__anext__())
                        tasks[i] = new_task
                        pending_tasks.add(new_task)
                        break
            except StopAsyncIteration:
                continue

class CustomSelectiveAgent10(BaseAgent):
    """Runs only selected sub-agents based on MainCoordinator output (from session state)."""

    _output_key: Optional[str] = PrivateAttr(default="sub_agent_outputs")

    def __init__(self, name: str, sub_agents: list[BaseAgent], output_key: Optional[str] = None):
        super().__init__(name=name, sub_agents=sub_agents)
        if output_key:
            self._output_key = output_key

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Get plan from session state
        plan = ctx.session.state.get("sub_agent_plan", "")

        try:
            task_list = plan if isinstance(plan, list) else json.loads(plan)
        except Exception as e:
            raise ValueError(f"Invalid JSON in sub_agent_plan: {e}")

        # Map agent name → query
        task_map = {task["agent"]: task["query"] for task in task_list if "agent" in task and "query" in task}

        if not task_map:
            raise ValueError("No valid sub-agent entries in plan.")

        # Chỉ chọn đúng các agent được giao task
        selected_agents = [agent for agent in self.sub_agents if agent.name in task_map]

        # Log để debug
        print(f"[CustomSelectiveAgent10] Will run {len(selected_agents)} sub-agents: {[agent.name for agent in selected_agents]}")
        for agent in selected_agents:
            print(f"  - {agent.name}: {task_map[agent.name]}")

        if not selected_agents:
            raise ValueError("No sub-agent matches the plan.")

        # Track output collection (optional)
        collected_outputs = {}

        # Chạy từng task với đúng agent và query (không group theo agent)
        agent_runs = []
        for task in task_list:
            agent_name = task["agent"]
            query = task["query"]
            agent = next((a for a in self.sub_agents if a.name == agent_name), None)
            if agent is None:
                print(f"[CustomSelectiveAgent10] Warning: Agent {agent_name} not found!")
                continue

            new_ctx = _create_branch_ctx_for_sub_agent(self, agent, ctx).copy(update={
                "input": query
            })

            async def track_output(agent=agent, new_ctx=new_ctx):
                async for event in agent.run_async(new_ctx):
                    if hasattr(event, 'content') and event.content:
                        collected_outputs.setdefault(agent.name, []).append(event.content)
                    yield event

            agent_runs.append(track_output())

        async for event in _merge_agent_run(agent_runs):
            yield event

        # Save outputs to session state if configured
        if self._output_key:
            ctx.session.state[self._output_key] = collected_outputs
