import logging
import json
import base64
import os
from typing import AsyncGenerator, List, Dict, Any
from typing_extensions import override
import asyncio
from google.adk.agents import LlmAgent, BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field

# Import agents và tools
from agent.agent import (
    main_agent,
    search_agents,
    vlm_agents,
    aggregator_agent,
    key_manager,
)

# --- Constants ---
APP_NAME = "cosmo_app"
USER_ID = "cosmo_user"
SESSION_ID = "cosmo_session"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Custom Orchestrator Agent ---
class CosmoFlowAgent(BaseAgent):  
    """  
    Custom agent for Cosmo workflow - tìm kiếm ảnh và phân tích VLM.  
  
    Flow: User Query → Main Agent → Search Agents (parallel) → VLM Agents (manual) → Aggregator Agent  
    """  
  
    # --- Field Declarations for Pydantic ---  
    main_agent: LlmAgent  
    parallel_search: ParallelAgent  
    aggregator_agent: LlmAgent  
      
    # Store original agents for reference  
    search_agents: List[LlmAgent]  
    vlm_agents: List[LlmAgent]  
  
    # model_config allows setting Pydantic configurations  
    model_config = {"arbitrary_types_allowed": True}  
  
    def __init__(  
        self,  
        name: str,  
        main_agent: LlmAgent,  
        search_agents: List[LlmAgent],  
        vlm_agents: List[LlmAgent],  
        aggregator_agent: LlmAgent,  
    ):  
        """  
        Initializes the CosmoFlowAgent.  
  
        Args:  
            name: The name of the agent.  
            main_agent: Agent để phân tích query và chia task  
            search_agents: List các Search Agents để tìm ảnh  
            vlm_agents: List các VLM Agents để phân tích ảnh  
            aggregator_agent: Agent để tổng hợp kết quả cuối cùng  
        """  
        # Create workflow agents *before* calling super().__init__  
        parallel_search = ParallelAgent(  
            name="ParallelSearchAgent",  
            sub_agents=search_agents  
        )  

        sub_agents_list = [  
            main_agent,  
            parallel_search,  
            aggregator_agent,  
        ]  
        super().__init__(  
            name=name,  
            main_agent=main_agent,  
            parallel_search=parallel_search,  
            aggregator_agent=aggregator_agent,  
            search_agents=search_agents,  
            vlm_agents=vlm_agents,        
            sub_agents=sub_agents_list,  
        )
    @override  
    async def _run_async_impl(  
        self, ctx: InvocationContext  
    ) -> AsyncGenerator[Event, None]:  
        """  
        Implements the custom orchestration logic for the Cosmo workflow.  
        Uses the workflow agents created during initialization.  
        """  
        logger.info(f"[{self.name}] Starting Cosmo workflow.")  
  
        # Lấy user query từ input content  
        user_input = self._extract_user_input(ctx)  
        logger.info(f"[{self.name}] User Query: {user_input}")  
  
        # Store user query in session  
        ctx.session.state["user_query"] = user_input  
  
        try:  
            # Step 1: Main Agent phân tích và chia task  
            logger.info(f"[{self.name}] Step 1: Main Agent analyzing query...")  
            async for event in self.main_agent.run_async(ctx):  
                yield event  
  
            # Check if task results were generated  
            if "task_results" not in ctx.session.state or not ctx.session.state["task_results"]:  
                logger.error(f"[{self.name}] Failed to generate Sub-Tasks. Aborting workflow.")  
                return  
  
            # Parse task results  
            task_results = ctx.session.state.get("task_results")  
            if isinstance(task_results, str):  
                search_tasks = json.loads(task_results)  
            else:  
                search_tasks = task_results  
  
            # Set up search queries for parallel search  
            for i, task in enumerate(search_tasks):  
                if i < len(self.search_agents):  
                    # Sử dụng key riêng cho từng agent  
                    ctx.session.state[f"search_query_{i+1}"] = task["query"]  
                    logger.info(f"[{self.name}] Set query for SearchAgent{i+1}: {task['query']}")
            # Step 2: Parallel Search Agents  
            
            logger.info(f"[{self.name}] Step 2: Search Agents working in parallel...")  
            search_results = []  
            async for event in self.parallel_search.run_async(ctx):  
                logger.info(f"[{self.name}] Search event: {event.author}")  
                  
                # Collect search results from function responses  
                if (hasattr(event, "content") and event.content and event.content.parts):  
                    for part in event.content.parts:  
                        if (hasattr(part, "function_response") and part.function_response):  
                            response_data = part.function_response.response  
                            if (isinstance(response_data, dict) and "images" in response_data):  
                                images = response_data["images"]  
                                search_results.extend(images)  
                                logger.info(f"[{self.name}] {event.author}: Found {len(images)} images")  
                  
                yield event  
  
            logger.info(f"[{self.name}] Total images found: {len(search_results)}")  
            
            # Step 3: VLM Agents phân tích ảnh song song
            logger.info(f"[{self.name}] Step 3: VLM Agents analyzing images...")
            
            if not search_results:
                logger.warning(f"[{self.name}] No images found, skipping VLM analysis")
                final_answer = "Xin lỗi, không tìm thấy hình ảnh liên quan đến câu hỏi của bạn."
            else:
                # Store images for VLM processing
                ctx.session.state["search_images"] = search_results
                
                # Chia ảnh cho các VLM agents
                image_batches = self._distribute_images_to_agents(search_results)
                
                all_vlm_results = []
                
                # Chạy VLM agents song song
                async def run_vlm_agent_batch(agent, images_batch, agent_idx):
                    batch_results = []
                    
                    for i, image in enumerate(images_batch):
                        try:
                            # Chuẩn bị input cho VLM agent với ảnh
                            vlm_input = await self._prepare_vlm_input_with_image(
                                user_input, image
                            )
                            
                            logger.info(
                                f"[{self.name}] {agent.name} processing image {image.get('id', f'img_{i}')}..."
                            )
                            
                            # Tạo content với cả text và ảnh
                            content_parts = [types.Part(text=vlm_input["text_input"])]
                            
                            # Thêm ảnh nếu có
                            if "image_part" in vlm_input:
                                content_parts.append(vlm_input["image_part"])
                                logger.info(
                                    f"[{self.name}] Added image to VLM input: {image.get('id')}"
                                )
                            
                            # Tạo content với ảnh
                            vlm_content = types.Content(role="user", parts=content_parts)
                            
                            # Tạo session service cho VLM nếu cần
                            vlm_session_service = InMemorySessionService()
                            
                            # Tạo runner cho VLM agent này
                            vlm_runner = Runner(
                                agent=agent,
                                app_name="CosmoVLM",
                                session_service=vlm_session_service,
                            )
                            
                            # Gọi VLM agent với ảnh - giới hạn events
                            vlm_response = ""
                            event_count = 0
                            max_vlm_events = 10  # Giới hạn cho VLM
                            
                            # Tạo session riêng cho VLM
                            vlm_session_id = f"vlm_{agent.name}_{image.get('id', 'unknown')}"
                            vlm_user_id = "vlm_user"
                            
                            # Tạo session trước
                            await vlm_session_service.create_session(
                                app_name="CosmoVLM",
                                user_id=vlm_user_id,
                                session_id=vlm_session_id,
                                state={},
                            )
                            
                            # Sử dụng runner để gọi agent với content có ảnh
                            events = vlm_runner.run_async(
                                user_id=vlm_user_id,
                                session_id=vlm_session_id,
                                new_message=vlm_content,
                            )
                            
                            async for event in events:
                                event_count += 1
                                if (
                                    hasattr(event, "content")
                                    and event.content
                                    and event.content.parts
                                ):
                                    if event.content.parts[0].text:
                                        vlm_response = event.content.parts[0].text
                                        break  # Break ngay khi có text response
                                
                                if event_count >= max_vlm_events:
                                    logger.warning(
                                        f"[{self.name}] {agent.name}: Reached max VLM events limit"
                                    )
                                    break
                            
                            batch_results.append(
                                {
                                    "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                                    "image_path": image.get("path", ""),
                                    "vlm_agent": agent.name,
                                    "response": vlm_response,
                                    "relevance_score": image.get("relevance_score", 0.0),
                                }
                            )
                            
                            logger.info(
                                f"[{self.name}] {image.get('id')}: {vlm_response[:50]}..."
                            )
                            
                        except Exception as e:
                            logger.error(
                                f"[{self.name}] Error processing {image.get('id', f'img_{i}')}: {e}"
                            )
                            batch_results.append(
                                {
                                    "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                                    "image_path": image.get("path", ""),
                                    "vlm_agent": agent.name,
                                    "response": f"Lỗi xử lý ảnh: {str(e)}",
                                    "relevance_score": image.get("relevance_score", 0.0),
                                }
                            )
                    
                    logger.info(
                        f"[{self.name}] {agent.name} completed batch: {len(batch_results)}/{len(images_batch)} processed"
                    )
                    return batch_results
                
                # Chạy tất cả VLM agents song song
                vlm_coroutines = [
                    run_vlm_agent_batch(self.vlm_agents[i], batch, i + 1)
                    for i, batch in enumerate(image_batches)
                    if batch
                ]
                
                vlm_results_list = await asyncio.gather(*vlm_coroutines, return_exceptions=True)
                
                # Tổng hợp kết quả
                for result in vlm_results_list:
                    if isinstance(result, Exception):
                        logger.error(f"[{self.name}] VLM batch failed: {result}")
                    elif isinstance(result, list):
                        all_vlm_results.extend(result)
                
                # Yield events để maintain ADK pattern
                for batch_result in all_vlm_results:
                    # Tạo event từ VLM result để yield
                    vlm_content = types.Content(
                        role="assistant",
                        parts=[types.Part(text=batch_result["response"])]
                    )
                    vlm_event = Event(
                        author=batch_result["vlm_agent"],
                        content=vlm_content
                    )
                    yield vlm_event
                
                # Store VLM results for aggregator
                ctx.session.state["vlm_results"] = all_vlm_results
                logger.info(
                    f"[{self.name}] VLM processing completed: {len(all_vlm_results)} total results"
                )
                # Step 4: Aggregator tổng hợp kết quả  
                logger.info(f"[{self.name}] Step 4: Aggregating final results...")  
                async for event in self.aggregator_agent.run_async(ctx):  
                    yield event  
  
                # Get final answer  
                final_answer = ctx.session.state.get("final_answer", "Không thể tạo câu trả lời cuối cùng.")  
  
            # Store final answer  
            ctx.session.state["final_answer"] = final_answer  
  
            # Yield final response event  
            final_content = types.Content(  
                role="assistant", parts=[types.Part(text=final_answer)]  
            )  
            final_event = Event(author=self.name, content=final_content)  
            yield final_event  
  
            logger.info(f"[{self.name}] Cosmo workflow completed successfully.")  
  
        except Exception as e:  
            logger.error(f"[{self.name}] Workflow error: {str(e)}")  
            error_content = types.Content(  
                role="assistant",  
                parts=[types.Part(text=f"Xin lỗi, có lỗi xảy ra: {str(e)}")],  
            )  
            error_event = Event(author=self.name, content=error_content)  
            yield error_event  
  
    def _extract_user_input(self, ctx: InvocationContext) -> str:  
        """Extract user input từ context"""  
        if hasattr(ctx, "request") and ctx.request and ctx.request.content:  
            if ctx.request.content.parts:  
                return ctx.request.content.parts[0].text  
        return ctx.session.state.get("user_query", "")  
  
    def _distribute_images_to_agents(
        self, all_images: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Chia ảnh cho các VLM agents
        """
        num_agents = len(self.vlm_agents)
        num_images = len(all_images)

        # Tính toán phân bổ
        base_batch_size = num_images // num_agents
        remainder = num_images % num_agents

        image_batches = []
        start_idx = 0

        for i in range(num_agents):
            batch_size = base_batch_size + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size

            batch = all_images[start_idx:end_idx]
            image_batches.append(batch)

            if batch:
                logger.info(
                    f"[{self.name}] {self.vlm_agents[i].name}: Assigned {len(batch)} images"
                )

            start_idx = end_idx

        return image_batches

    async def _prepare_vlm_input_with_image(
        self, user_query: str, image: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Chuẩn bị input cho VLM agent - bao gồm text và ảnh theo Google ADK format
        """
        # Text input cho VLM
        text_input = f"""User query: {user_query}
Image ID: {image.get('id', 'unknown')}
Image description: {image.get('description', '')}
Relevance score: {image.get('relevance_score', 0.0)}

Please analyze this image and answer the user's question."""

        vlm_input = {
            "text_input": text_input,
            "image_id": image.get("id", "unknown"),
            "user_query": user_query,
        }

        # Xử lý ảnh - đọc và encode base64 cho Google ADK
        image_path = image.get("path")
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    # Google ADK format cho ảnh
                    image_mime_type = "image/png"  # hoặc detect từ extension
                    vlm_input["image_part"] = types.Part(
                        inline_data=types.Blob(
                            mime_type=image_mime_type, data=image_data
                        )
                    )

                logger.info(
                    f"[{self.name}] Image prepared: {image_path} ({len(image_data)} bytes)"
                )

            except Exception as e:
                logger.error(f"[{self.name}] Error reading image {image_path}: {e}")
                vlm_input["image_error"] = f"Cannot read image: {e}"
        else:
            logger.warning(f"[{self.name}] Image path not found: {image_path}")
            vlm_input["image_error"] = "Image file not found"

        return vlm_input  
  
  
# --- Create the CosmoFlowAgent instance ---  
cosmo_flow_agent = CosmoFlowAgent(  
    name="CosmoFlowAgent",  
    main_agent=main_agent,  
    search_agents=search_agents,  
    vlm_agents=vlm_agents,  
    aggregator_agent=aggregator_agent,  
)  
  
logger.info(  
    f"✅ CosmoFlowAgent initialized with {len(search_agents)} search agents and {len(vlm_agents)} VLM agents"  
)