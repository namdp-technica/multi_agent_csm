import logging
import json
from typing import AsyncGenerator, List
from typing_extensions import override
import asyncio
from google.adk.agents import LlmAgent, BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.events import Event
from pydantic import BaseModel
from utils.helper_workflow import (
    extract_user_input,
)
from workflow.vlm_runner import run_all_vlm_batches
# Import agents và tools
from agent.agent import (
    search_agents,
    vlm_agents,
)
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
        user_input = extract_user_input(ctx)  
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
                
                # Sử dụng VLM runner để xử lý ảnh song song
                all_vlm_results = await run_all_vlm_batches(user_input, search_results, self.vlm_agents)
                
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
logger.info(  
    f"✅ CosmoFlowAgent initialized with {len(search_agents)} search agents and {len(vlm_agents)} VLM agents"  
)