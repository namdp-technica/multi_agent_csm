import logging
import json
import base64
import os
from typing import AsyncGenerator, List, Dict, Any
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent
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
    key_manager
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

    Flow: User Query → Main Agent → Search Agents (parallel) → VLM Agents → Aggregator Agent
    """

    # --- Field Declarations for Pydantic ---
    main_agent: LlmAgent
    search_agents: List[LlmAgent]
    vlm_agents: List[LlmAgent]
    aggregator_agent: LlmAgent

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
        # Define the sub_agents list for the framework
        sub_agents_list = [main_agent] + search_agents + vlm_agents + [aggregator_agent]

        # Pydantic will validate and assign them based on the class annotations.
        super().__init__(
            name=name,
            main_agent=main_agent,
            search_agents=search_agents,
            vlm_agents=vlm_agents,
            aggregator_agent=aggregator_agent,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the Cosmo workflow.
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
            search_tasks = await self._step1_main_agent_analysis_sync(ctx, user_input)

            # Step 2: Search Agents tìm kiếm ảnh song song
            logger.info(f"[{self.name}] Step 2: Search Agents working in parallel...")
            all_images = await self._step2_parallel_search_sync(ctx, search_tasks, user_input)

            # Step 3: VLM Agents phân tích ảnh
            logger.info(f"[{self.name}] Step 3: VLM Agents analyzing images...")  
            vlm_results = await self._step3_vlm_analysis_sync(ctx, user_input, all_images)

            # Step 4: Aggregator tổng hợp kết quả
            logger.info(f"[{self.name}] Step 4: Aggregating final results...")
            final_answer = await self._step4_aggregate_results_sync(ctx, user_input, vlm_results)

            # Store final answer
            ctx.session.state["final_answer"] = final_answer
            
            # Yield final response event
            final_content = types.Content(
                role='assistant',
                parts=[types.Part(text=final_answer)]
            )
            
            final_event = Event(
                author=self.name,
                content=final_content
            )
            
            yield final_event
            
            logger.info(f"[{self.name}] Cosmo workflow completed successfully.")

        except Exception as e:
            logger.error(f"[{self.name}] Workflow error: {str(e)}")
            error_content = types.Content(
                role='assistant',
                parts=[types.Part(text=f"Xin lỗi, có lỗi xảy ra: {str(e)}")]
            )
            error_event = Event(
                author=self.name,
                content=error_content
            )
            yield error_event

    def _extract_user_input(self, ctx: InvocationContext) -> str:
        """Extract user input từ context"""
        if hasattr(ctx, 'request') and ctx.request and ctx.request.content:
            if ctx.request.content.parts:
                return ctx.request.content.parts[0].text
        return ctx.session.state.get("user_query", "")

    async def _step1_main_agent_analysis_sync(self, ctx: InvocationContext, user_query: str) -> List[Dict[str, str]]:
        """
        Step 1: Main Agent phân tích câu hỏi và chia thành sub-tasks
        """
        logger.info(f"[{self.name}] Main Agent analyzing and breaking down the query...")
        
        try:
            # Gọi main agent - giới hạn events
            main_response = ""
            event_count = 0
            max_main_events = 5  # Giới hạn cho Main Agent
            
            async for event in self.main_agent.run_async(ctx):
                event_count += 1
                logger.info(f"[{self.name}] Main Agent Event #{event_count}: {event.model_dump_json(indent=2, exclude_none=True)}")
                if hasattr(event, 'content') and event.content and event.content.parts:
                    if event.content.parts[0].text:
                        main_response = event.content.parts[0].text
                        break  # Break ngay khi có response
                
                if event_count >= max_main_events:
                    logger.warning(f"[{self.name}] Main Agent: Reached max events limit")
                    break

            # Parse JSON response từ main agent output
            main_output = main_response 
            if isinstance(main_output, str):
                search_tasks_str = main_output
            else:
                search_tasks_str = str(main_output)
            
            logger.info(f"[{self.name}] Raw main agent response: {search_tasks_str}")
            
            # Clean và parse JSON
            search_tasks_str = search_tasks_str.strip()
            if search_tasks_str.startswith('```json'):
                search_tasks_str = search_tasks_str[7:-3]
            elif search_tasks_str.startswith('```'):
                search_tasks_str = search_tasks_str[3:-3]
                
            search_tasks = json.loads(search_tasks_str)
            
            logger.info(f"[{self.name}] Created {len(search_tasks)} search tasks")
            for i, task in enumerate(search_tasks, 1):
                logger.info(f"[{self.name}]   {i}. {task['agent']}: {task['query']}")
                
            return search_tasks
            
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] JSON parsing error: {e}")
            # Fallback: tạo search tasks mặc định
            return [
                {"agent": "SearchAgent1", "query": user_query},
                {"agent": "SearchAgent2", "query": f"related to {user_query}"},
                {"agent": "SearchAgent3", "query": f"information about {user_query}"}
            ]
        except Exception as e:
            logger.error(f"[{self.name}] Main agent error: {e}")
            raise

    async def _step2_parallel_search_sync(self, ctx: InvocationContext, search_tasks: List[Dict[str, str]], user_query: str) -> List[Dict[str, Any]]:
        """
        Step 2: Chạy 3 Search Agents song song
        """
        logger.info(f"[{self.name}] Running {len(self.search_agents)} Search Agents in parallel...")
        
        all_images = []
        
        # Chạy search agents song song  
        import asyncio
        async def run_search_agent(agent, task, agent_idx):
            try:
                # Create input cho search agent - chỉ query đơn giản
                ctx.session.state["search_query"] = task['query']
                
                # Gọi search agent với input đơn giản
                search_results = []
                event_count = 0
                max_events = 20  # Giới hạn số events để tránh lặp vô tận
                
                async for event in agent.run_async(ctx):
                    event_count += 1
                    logger.info(f"[{self.name}] SearchAgent{agent_idx} Event #{event_count}: {event.model_dump_json(indent=2, exclude_none=True)}")
                    
                    # Kiểm tra function response có chứa search results
                    if (hasattr(event, 'content') and event.content and event.content.parts):
                        for part in event.content.parts:
                            if hasattr(part, 'function_response') and part.function_response:
                                response_data = part.function_response.response
                                if isinstance(response_data, dict) and "images" in response_data:
                                    search_results = response_data["images"]
                                    logger.info(f"[{self.name}] SearchAgent{agent_idx}: Found {len(search_results)} images in function response")
                                    return search_results  # Return ngay khi có kết quả
                    
                    # Giới hạn số events để tránh lặp vô tận
                    if event_count >= max_events:
                        logger.warning(f"[{self.name}] SearchAgent{agent_idx}: Reached max events limit ({max_events})")
                        break
                        
                logger.info(f"[{self.name}] SearchAgent{agent_idx}: Found {len(search_results)} images total")
                return search_results
                
            except Exception as e:
                logger.error(f"[{self.name}] SearchAgent{agent_idx} error: {e}")
                return []
        
        # Chạy tất cả search agents song song
        search_tasks_limited = search_tasks[:len(self.search_agents)]  # Giới hạn số task
        search_coroutines = [
            run_search_agent(self.search_agents[i], task, i+1) 
            for i, task in enumerate(search_tasks_limited)
        ]
        
        search_results_list = await asyncio.gather(*search_coroutines, return_exceptions=True)
        
        # Tổng hợp kết quả
        for i, result in enumerate(search_results_list):
            if isinstance(result, Exception):
                logger.error(f"[{self.name}] SearchAgent{i+1} failed: {result}")
            elif isinstance(result, list):
                all_images.extend(result)
        
        logger.info(f"[{self.name}] Total images found: {len(all_images)}")
        return all_images

    async def _step3_vlm_analysis_sync(self, ctx: InvocationContext, user_query: str, all_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: VLM Agents phân tích ảnh song song
        """
        logger.info(f"[{self.name}] VLM Agents processing {len(all_images)} images in parallel...")
        
        if not all_images:
            logger.warning(f"[{self.name}] No images to process")
            return []
        
        # Chia ảnh cho các VLM agents
        image_batches = self._distribute_images_to_agents(all_images)
        
        all_vlm_results = []
        
        # Chạy VLM agents song song
        import asyncio
        async def run_vlm_agent_batch(agent, images_batch, agent_idx):
            batch_results = []
            
            for i, image in enumerate(images_batch):
                try:
                    # Chuẩn bị input cho VLM agent với ảnh
                    vlm_input = await self._prepare_vlm_input_with_image(user_query, image)
                    
                    logger.info(f"[{self.name}] {agent.name} processing image {image.get('id', f'img_{i}')}...")
                    
                    # Tạo content với cả text và ảnh
                    content_parts = [types.Part(text=vlm_input["text_input"])]
                    
                    # Thêm ảnh nếu có
                    if "image_part" in vlm_input:
                        content_parts.append(vlm_input["image_part"])
                        logger.info(f"[{self.name}] Added image to VLM input: {image.get('id')}")
                    
                    # Tạo content với ảnh
                    vlm_content = types.Content(
                        role='user',
                        parts=content_parts
                    )
                    
                    # Tạo session runner cho VLM agent với ảnh
                    from google.adk.runners import Runner
                    from google.adk.sessions import InMemorySessionService
                    
                    # Tạo session service cho VLM nếu cần
                    vlm_session_service = InMemorySessionService()
                    
                    # Tạo runner cho VLM agent này
                    vlm_runner = Runner(
                        agent=agent,
                        app_name="CosmoVLM",
                        session_service=vlm_session_service
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
                        state={}
                    )
                    
                    # Sử dụng runner để gọi agent với content có ảnh
                    events = vlm_runner.run_async(
                        user_id=vlm_user_id,
                        session_id=vlm_session_id,
                        new_message=vlm_content
                    )
                    
                    async for event in events:
                        event_count += 1
                        if hasattr(event, 'content') and event.content and event.content.parts:
                            if event.content.parts[0].text:
                                vlm_response = event.content.parts[0].text
                                break  # Break ngay khi có text response
                        
                        if event_count >= max_vlm_events:
                            logger.warning(f"[{self.name}] {agent.name}: Reached max VLM events limit")
                            break
                    
                    batch_results.append({
                        "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                        "image_path": image.get("path", ""),
                        "vlm_agent": agent.name,
                        "response": vlm_response,
                        "relevance_score": image.get("relevance_score", 0.0)
                    })
                    
                    logger.info(f"[{self.name}] {image.get('id')}: {vlm_response[:50]}...")
                    
                except Exception as e:
                    logger.error(f"[{self.name}] Error processing {image.get('id', f'img_{i}')}: {e}")
                    batch_results.append({
                        "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                        "image_path": image.get("path", ""),
                        "vlm_agent": agent.name,
                        "response": f"Lỗi xử lý ảnh: {str(e)}",
                        "relevance_score": image.get("relevance_score", 0.0)
                    })
            
            logger.info(f"[{self.name}] {agent.name} completed batch: {len(batch_results)}/{len(images_batch)} processed")
            return batch_results
        
        # Chạy tất cả VLM agents song song
        vlm_coroutines = [
            run_vlm_agent_batch(self.vlm_agents[i], batch, i+1) 
            for i, batch in enumerate(image_batches) if batch
        ]
        
        vlm_results_list = await asyncio.gather(*vlm_coroutines, return_exceptions=True)
        
        # Tổng hợp kết quả
        for result in vlm_results_list:
            if isinstance(result, Exception):
                logger.error(f"[{self.name}] VLM batch failed: {result}")
            elif isinstance(result, list):
                all_vlm_results.extend(result)
        
        logger.info(f"[{self.name}] VLM processing completed: {len(all_vlm_results)} total results")
        return all_vlm_results

    def _distribute_images_to_agents(self, all_images: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
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
                logger.info(f"[{self.name}] {self.vlm_agents[i].name}: Assigned {len(batch)} images")
            
            start_idx = end_idx
        
        return image_batches

    async def _prepare_vlm_input_with_image(self, user_query: str, image: Dict[str, Any]) -> Dict[str, Any]:
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
            "user_query": user_query
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
                            mime_type=image_mime_type,
                            data=image_data
                        )
                    )
                    
                logger.info(f"[{self.name}] Image prepared: {image_path} ({len(image_data)} bytes)")
                
            except Exception as e:
                logger.error(f"[{self.name}] Error reading image {image_path}: {e}")
                vlm_input["image_error"] = f"Cannot read image: {e}"
        else:
            logger.warning(f"[{self.name}] Image path not found: {image_path}")
            vlm_input["image_error"] = "Image file not found"
        
        return vlm_input

    async def _step4_aggregate_results_sync(self, ctx: InvocationContext, user_query: str, vlm_results: List[Dict[str, Any]]) -> str:
        """
        Step 4: Aggregator tổng hợp kết quả cuối cùng
        """
        logger.info(f"[{self.name}] Aggregator processing final results...")
        
        try:
            # Chuẩn bị input cho aggregator
            aggregator_input = f"""User query: {user_query}
Total VLM results: {len(vlm_results)}

VLM Analysis Results:
{json.dumps(vlm_results, indent=2, ensure_ascii=False)}

Please provide a comprehensive answer based on the VLM analysis results."""
            
            # Set input vào session
            ctx.session.state["aggregator_input"] = aggregator_input
            
            # Store VLM results trong session state để aggregator có thể truy cập
            ctx.session.state["vlm_results"] = vlm_results
            ctx.session.state["user_query"] = user_query
            
            # Gọi aggregator agent trực tiếp với aggregator_input
            from google.genai import types
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            
            # Tạo content mới với aggregator_input
            aggregator_content = types.Content(
                role='user',
                parts=[types.Part(text=aggregator_input)]
            )
            
            # Tạo runner mới cho aggregator với content chứa VLM results
            session_service = InMemorySessionService()
            temp_session = await session_service.create_session(
                app_name="cosmo_temp", 
                user_id="temp_user", 
                session_id="temp_session"
            )
            
            runner = Runner(
                agent=self.aggregator_agent,
                app_name="cosmo_temp",
                session_service=session_service
            )
            
            # Gọi aggregator với content chứa VLM results
            final_answer = ""
            event_count = 0
            max_aggregator_events = 5  # Giới hạn cho Aggregator
            
            events = runner.run_async(
                user_id="temp_user",
                session_id="temp_session", 
                new_message=aggregator_content
            )
            
            async for event in events:
                event_count += 1
                logger.info(f"[{self.name}] Aggregator Event #{event_count}: {event.model_dump_json(indent=2, exclude_none=True)}")
                if hasattr(event, 'content') and event.content and event.content.parts:
                    if event.content.parts[0].text:
                        final_answer = event.content.parts[0].text
                        break  # Break ngay khi có final answer
                
                if event_count >= max_aggregator_events:
                    logger.warning(f"[{self.name}] Aggregator: Reached max events limit")
                    break
            
            logger.info(f"[{self.name}] Final answer generated ({len(final_answer)} chars)")
            return final_answer
            
        except Exception as e:
            logger.error(f"[{self.name}] Aggregator error: {e}")
            # Fallback: tự tổng hợp
            return self._fallback_aggregation(user_query, vlm_results)

    def _fallback_aggregation(self, user_query: str, vlm_results: List[Dict[str, Any]]) -> str:
        """
        Tổng hợp dự phòng khi aggregator lỗi
        """
        if not vlm_results:
            return "Xin lỗi, không tìm thấy thông tin liên quan đến câu hỏi của bạn."
        
        # Lọc các kết quả có ý nghĩa
        meaningful_results = [
            result for result in vlm_results 
            if result["response"] and "không biết" not in result["response"].lower()
        ]
        
        if not meaningful_results:
            return "Tôi không thể tìm thấy thông tin phù hợp để trả lời câu hỏi này."
        
        # Tổng hợp câu trả lời
        answer_parts = [f"Dựa trên {len(meaningful_results)} ảnh được phân tích:"]
        
        for i, result in enumerate(meaningful_results[:3], 1):  # Lấy 3 kết quả tốt nhất
            answer_parts.append(f"{i}. {result['response']} (từ {result['image_id']})")
        
        return "\n".join(answer_parts)


# --- Create the CosmoFlowAgent instance ---
cosmo_flow_agent = CosmoFlowAgent(
    name="CosmoFlowAgent",
    main_agent=main_agent,
    search_agents=search_agents,
    vlm_agents=vlm_agents,
    aggregator_agent=aggregator_agent,
)

logger.info(f"✅ CosmoFlowAgent initialized with {len(search_agents)} search agents and {len(vlm_agents)} VLM agents")