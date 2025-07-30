import json
import time
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any
from pydantic import PrivateAttr
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext

def _create_branch_ctx_for_vlm_agent(
    parent_agent: BaseAgent,
    vlm_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
    """Tạo context riêng cho mỗi VLM agent."""
    invocation_context = invocation_context.model_copy()
    branch_suffix = f"{parent_agent.name}.{vlm_agent.name}"
    invocation_context.branch = (
        f"{invocation_context.branch}.{branch_suffix}"
        if invocation_context.branch
        else branch_suffix
    )
    return invocation_context

async def _merge_vlm_agent_runs(agent_runs: list[AsyncGenerator[Event, None]]) -> AsyncGenerator[Event, None]:
    """Gộp kết quả từ tất cả VLM agents chạy song song."""
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

class VLMWorkflow(BaseAgent):
    """Workflow chạy K VLM agents song song để phân tích ảnh và trả lời câu hỏi"""

    _output_key: Optional[str] = PrivateAttr(default="vlm_responses")

    def __init__(self, name: str, vlm_agents: List[LlmAgent], output_key: Optional[str] = None):
        super().__init__(name=name, sub_agents=vlm_agents)
        
        if output_key:
            self._output_key = output_key

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Lấy danh sách ảnh từ retriever
        retrieved_images = ctx.session.state.get("retrieved_images", {})
        user_query = ctx.session.state.get("user_query", "")
        
        if not retrieved_images or "images" not in retrieved_images:
            print("[VLMWorkflow] No images found from retriever!")
            return
            
        images = retrieved_images["images"]
        print(f"[VLMWorkflow] Processing {len(images)} images with {len(self.sub_agents)} VLM agents")
        
        # Phân chia ảnh cho các VLM agents
        vlm_responses = {}
        agent_runs = []
        
        for i, vlm_agent in enumerate(self.sub_agents):
            # Mỗi agent sẽ được phân 1 ảnh (hoặc None nếu không đủ ảnh)
            assigned_image = images[i] if i < len(images) else None
            
            if assigned_image is None:
                print(f"[VLMWorkflow] No image assigned to {vlm_agent.name}")
                continue
                
            print(f"[VLMWorkflow] Assigning image {assigned_image['id']} to {vlm_agent.name}")
            
            # Tạo context riêng cho VLM agent
            new_ctx = _create_branch_ctx_for_vlm_agent(self, vlm_agent, ctx).copy(update={
                "input": f"Analyze this image and answer the question: {user_query}",
                "assigned_image": assigned_image
            })

            async def track_vlm_output(agent=vlm_agent, new_ctx=new_ctx, assigned_image=assigned_image):
                async for event in agent.run_async(new_ctx):
                    if hasattr(event, 'content') and event.content:
                        vlm_responses[agent.name] = {
                            "response": event.content,
                            "image_id": assigned_image["id"],
                            "image_description": assigned_image.get("description", "")
                        }
                    yield event

            agent_runs.append(track_vlm_output())

        # Chạy tất cả VLM agents song song
        async for event in _merge_vlm_agent_runs(agent_runs):
            yield event

        # Lưu kết quả vào session state
        if self._output_key:
            ctx.session.state[self._output_key] = vlm_responses
            print(f"[VLMWorkflow] Saved {len(vlm_responses)} VLM responses to session state")

class CosmoWorkflow(BaseAgent):
    """
    Main VLM Workflow: Main Coordinator -> Retriever -> VLM Agents -> Final Response
    """
    
    def __init__(
        self,
        name: str,
        main_agent: LlmAgent,
        retriever_agent: LlmAgent,
        vlm_agents: List[LlmAgent],
        final_response_agent: LlmAgent
    ):
        # Tạo VLM workflow
        vlm_workflow = VLMWorkflow(
            name="VLMWorkflow", 
            vlm_agents=vlm_agents,
            output_key="vlm_responses"
        )
        
        # Tạo sequential workflow
        sequential_workflow = SequentialAgent(
            name="MainWorkflow",
            sub_agents=[
                main_agent,           # Bước 1: Nhận query từ user
                retriever_agent,      # Bước 2: Tìm kiếm ảnh
                vlm_workflow,         # Bước 3: Chạy K VLM agents song song
                final_response_agent  # Bước 4: Tổng hợp và trả lời cuối cùng
            ]
        )
        
        super().__init__(name=name, sub_agents=[sequential_workflow])
        self.sequential_workflow = sequential_workflow

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Chạy workflow chính"""
        print(f"[{self.name}] Starting VLM workflow...")
        
        start_time = time.time()
        
        # Chạy sequential workflow
        async for event in self.sequential_workflow.run_async(ctx):
            yield event
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"[{self.name}] VLM workflow completed in {execution_time:.2f}s")
        
        # Lưu thời gian thực hiện
        ctx.session.state["workflow_execution_time"] = execution_time

    
