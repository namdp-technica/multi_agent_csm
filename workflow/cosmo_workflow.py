import json
import time
import asyncio
import base64
import os
from typing import AsyncGenerator, Optional, List, Dict, Any
from pydantic import PrivateAttr
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

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

class SearchWorkflow(BaseAgent):
    """Workflow chạy multiple Search agents song song để tìm ảnh"""

    _output_key: Optional[str] = PrivateAttr(default="all_retrieved_images")

    def __init__(self, name: str, search_agents: List[LlmAgent], output_key: Optional[str] = None):
        super().__init__(name=name, sub_agents=search_agents)
        
        if output_key:
            self._output_key = output_key

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Lấy task list từ main agent
        task_plan = ctx.session.state.get("user_query", "")
        
        try:
            if isinstance(task_plan, str):
                # Remove markdown code blocks if present
                clean_json = task_plan.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:]  # Remove ```json
                if clean_json.endswith('```'):
                    clean_json = clean_json[:-3]  # Remove ```
                clean_json = clean_json.strip()
                task_list = json.loads(clean_json)
            else:
                task_list = task_plan
        except Exception as e:
            print(f"[SearchWorkflow] Error parsing task plan: {e}")
            print(f"[SearchWorkflow] Raw task_plan: {task_plan}")
            return
            
        if not isinstance(task_list, list):
            print(f"[SearchWorkflow] Invalid task list format")
            return
            
        print(f"[SearchWorkflow] Running {len(task_list)} search tasks with {len(self.sub_agents)} search agents")
        
        # Map task với agent
        search_responses = {}
        agent_runs = []
        
        for i, task in enumerate(task_list):
            if i >= len(self.sub_agents):
                break
                
            search_agent = self.sub_agents[i]
            query = task.get("query", "")
            
            print(f"[SearchWorkflow] Assigning query '{query}' to {search_agent.name}")
            
            new_ctx = _create_branch_ctx_for_vlm_agent(self, search_agent, ctx).copy(update={
                "input": f"Tìm kiếm ảnh cho: {query}"
            })

            async def track_search_output(agent=search_agent, new_ctx=new_ctx, query=query):
                async for event in agent.run_async(new_ctx):
                    if hasattr(event, 'content') and event.content:
                        search_responses[agent.name] = {
                            "response": event.content,
                            "query": query
                        }
                    yield event

            agent_runs.append(track_search_output())

        # Chạy tất cả Search agents song song
        async for event in _merge_vlm_agent_runs(agent_runs):
            yield event

        # Gom tất cả ảnh từ các search agents
        all_images = []
        for agent_name, response in search_responses.items():
            print(f"[SearchWorkflow] Processing response from {agent_name}")
            # Search trong session state cho function responses của agent này
            agent_images = []
            
            # Tìm trong session events cho function responses
            for event in ctx.session.events:
                if hasattr(event, 'author') and event.author == agent_name:
                    if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'function_response') and part.function_response:
                                if part.function_response.name == 'image_search':
                                    response_data = part.function_response.response
                                    if isinstance(response_data, dict) and 'images' in response_data:
                                        images = response_data['images']
                                        agent_images.extend(images)
                                        print(f"[SearchWorkflow] Got {len(images)} images from {agent_name}")
            
            all_images.extend(agent_images)

        # Lưu tất cả ảnh vào session state
        if self._output_key:
            ctx.session.state[self._output_key] = {
                "images": all_images,
                "total_found": len(all_images),
                "search_metadata": {
                    "search_agents_used": len(search_responses),
                    "queries": [resp["query"] for resp in search_responses.values()]
                }
            }
            print(f"[SearchWorkflow] Saved {len(all_images)} total images to session state")

class VLMWorkflow(BaseAgent):
    """Workflow chạy K VLM agents song song để phân tích ảnh và trả lời câu hỏi"""

    _output_key: Optional[str] = PrivateAttr(default="vlm_responses")

    def __init__(self, name: str, vlm_agents: List[LlmAgent], output_key: Optional[str] = None):
        super().__init__(name=name, sub_agents=vlm_agents)
        
        if output_key:
            self._output_key = output_key

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Lấy tất cả ảnh từ search workflow
        all_retrieved_images = ctx.session.state.get("all_retrieved_images", {})
        user_query = ctx.session.state.get("original_user_query", "")
        
        if not all_retrieved_images or "images" not in all_retrieved_images:
            print("[VLMWorkflow] No images found from search agents!")
            # Yield a message event instead of returning
            from google.genai.types import Event
            yield Event(content="[VLMWorkflow] No images found - skipping VLM analysis")
            return
            
        images = all_retrieved_images["images"]
        if not images or len(images) == 0:
            print("[VLMWorkflow] Empty images list from search agents!")
            # Yield a message event instead of returning  
            from google.genai.types import Event
            yield Event(content="[VLMWorkflow] Empty images list - skipping VLM analysis")
            return
            
        print(f"[VLMWorkflow] Processing {len(images)} images with {len(self.sub_agents)} VLM agents")
        
        # Phân chia ảnh cho các VLM agents (phân đều tất cả ảnh)
        vlm_responses = {}
        agent_runs = []
        
        # Phân chia ảnh đều cho các VLM agents
        images_per_agent = len(images) // len(self.sub_agents)
        remaining_images = len(images) % len(self.sub_agents)
        
        print(f"[VLMWorkflow] Distribution: {len(images)} images / {len(self.sub_agents)} agents = {images_per_agent} per agent, {remaining_images} remaining")
        
        image_index = 0
        for i, vlm_agent in enumerate(self.sub_agents):
            # Tính số ảnh cho agent này
            num_images_for_agent = images_per_agent
            if i < remaining_images:  # Phân ảnh dư cho các agent đầu tiên
                num_images_for_agent += 1
            
            if num_images_for_agent == 0:
                print(f"[VLMWorkflow] No images assigned to {vlm_agent.name}")
                continue
            
            # Lấy ảnh cho agent này
            agent_images = images[image_index:image_index + num_images_for_agent]
            image_index += num_images_for_agent
            
            print(f"[VLMWorkflow] Agent {i+1} ({vlm_agent.name}): {num_images_for_agent} images (index {image_index-num_images_for_agent} to {image_index-1})")
            print(f"[VLMWorkflow] Image IDs for {vlm_agent.name}: {[img['id'] for img in agent_images]}")
            
            # Tạo content với tất cả ảnh cho agent này
            text_content = f"""
Câu hỏi gốc từ user: {user_query}

Bạn được giao {len(agent_images)} ảnh để phân tích:
"""
            
            # Thêm thông tin từng ảnh
            for j, img in enumerate(agent_images):
                text_content += f"""
Ảnh {j+1}:
- ID: {img['id']}
- Đường dẫn: {img['path']}
- Mô tả: {img['description']}
- Điểm relevance: {img['relevance_score']}
"""
            
            text_content += f"\n\nHãy phân tích tất cả ảnh được giao và trả lời câu hỏi gốc của user dựa trên nội dung ảnh."
            
            # DEBUG: Kiểm tra xem có ảnh nào để load không
            valid_images = []
            for assigned_image in agent_images:
                image_path = assigned_image['path']
                if os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                        
                        # Encode base64 để debug
                        image_b64 = base64.b64encode(image_data).decode('utf-8')
                        
                        valid_images.append({
                            'id': assigned_image['id'],
                            'path': image_path,
                            'data': image_data,
                            'size': len(image_data)
                        })
                        print(f"[VLMWorkflow] Loaded image {assigned_image['id']}: {len(image_data)} bytes")
                    except Exception as e:
                        print(f"[VLMWorkflow] Error loading image {image_path}: {e}")
                else:
                    print(f"[VLMWorkflow] Image file not found: {image_path}")
            
            if not valid_images:
                print(f"[VLMWorkflow] WARNING: No valid images for {vlm_agent.name}!")
                # Vẫn tạo context với text thôi
                new_ctx = _create_branch_ctx_for_vlm_agent(self, vlm_agent, ctx).copy(update={
                    "input": text_content + "\n\n[ERROR: Không có ảnh hợp lệ để phân tích]",
                    "assigned_images": agent_images
                })
            else:
                # Tạo multimodal content với Gemini format
                content_parts = [types.Part(text=text_content)]
                
                # Thêm từng ảnh vào content parts
                for img in valid_images:
                    # Xác định mime type
                    if img['path'].lower().endswith('.png'):
                        mime_type = "image/png"
                    elif img['path'].lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    else:
                        mime_type = "image/png"
                    
                    content_parts.append(types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=img['data']
                        )
                    ))
                    print(f"[VLMWorkflow] Added {img['id']} ({mime_type}, {img['size']} bytes) to {vlm_agent.name}")
                
                # Tạo complete multimodal content
                multimodal_content = types.Content(
                    role='user',
                    parts=content_parts
                )
                
                print(f"[VLMWorkflow] Created multimodal content: {len(content_parts)} parts (1 text + {len(valid_images)} images)")
                
                new_ctx = _create_branch_ctx_for_vlm_agent(self, vlm_agent, ctx).copy(update={
                    "input": multimodal_content,  # Truyền complete multimodal content
                    "assigned_images": agent_images,
                    "image_count": len(valid_images)
                })

            async def track_vlm_output(agent=vlm_agent, new_ctx=new_ctx, agent_images=agent_images):
                async for event in agent.run_async(new_ctx):
                    if hasattr(event, 'content') and event.content:
                        vlm_responses[agent.name] = {
                            "response": event.content,
                            "image_ids": [img["id"] for img in agent_images],
                            "num_images": len(agent_images)
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
    New VLM Workflow: Main Agent -> Search Workflow -> VLM Workflow -> Aggregator
    """
    
    def __init__(
        self,
        name: str,
        main_agent: LlmAgent,
        search_agents: List[LlmAgent],
        vlm_agents: List[LlmAgent],
        aggregator_agent: LlmAgent
    ):
        # Store agents for internal use but DON'T pass to BaseAgent
        self._main_agent = main_agent
        self._search_agents = search_agents  
        self._vlm_agents = vlm_agents
        self._aggregator_agent = aggregator_agent
        # Tạo search workflow
        search_workflow = SearchWorkflow(
            name="SearchWorkflow", 
            search_agents=self._search_agents,
            output_key="all_retrieved_images"
        )
        
        # Tạo VLM workflow
        vlm_workflow = VLMWorkflow(
            name="VLMWorkflow", 
            vlm_agents=self._vlm_agents,
            output_key="vlm_responses"
        )
        
        # Tạo sequential workflow
        sequential_workflow = SequentialAgent(
            name="MainWorkflow",
            sub_agents=[
                self._main_agent,           # Bước 1: Phân tích và chia task
                search_workflow,            # Bước 2: Chạy 2-3 Search agents song song
                vlm_workflow,               # Bước 3: Chạy K VLM agents với tất cả ảnh
                self._aggregator_agent      # Bước 4: Tổng hợp kết quả cuối cùng
            ]
        )
        
        # ONLY pass sequential_workflow as sub_agent to BaseAgent
        super().__init__(name=name, sub_agents=[sequential_workflow])

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Chạy workflow chính"""
        print(f"[{self.name}] Starting New VLM workflow...")
        
        # Lưu câu hỏi gốc để VLM agents sử dụng
        original_query = None
        for event in ctx.session.events:
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        original_query = part.text
                        break
                if original_query:
                    break
        
        if original_query:
            ctx.session.state["original_user_query"] = original_query
            print(f"[{self.name}] Saved original query: {original_query}")
        
        start_time = time.time()
        
        # Chạy sequential workflow (lấy từ sub_agents[0])
        async for event in self.sub_agents[0].run_async(ctx):
            yield event
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"[{self.name}] New VLM workflow completed in {execution_time:.2f}s")
        
        # Lưu thời gian thực hiện
        ctx.session.state["workflow_execution_time"] = execution_time

    
