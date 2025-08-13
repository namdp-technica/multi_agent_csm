import logging
import json
import base64
import os
import asyncio
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

class CosmoFlowAgent(BaseAgent):
    """"""
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
        logger.info(f"[{self.name}] Starting Cosmo work flow")
        try: 

        #Initial Main Agent
            logger.info(f"[{self.name}] Step1 : Main Agent break task")
            await self._step1_main_agent_analysis_sync(ctx)

            await self._step2_parallel_search_sync(ctx)

            await self._step3_vlm_analysis_sync(ctx)  

            await self._step4_aggregate_results_sync(ctx)
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

    async def _step1_main_agent_analysis_sync(self, ctx: InvocationContext) -> List[Dict[str, str]]:
        try: 
            async for event in self.main_agent.run_async(ctx):
                yield event

            if "task_results" not in ctx.session.state or not ctx.session.state["task_results"]:
                logger.error(f"[{self.name}] Failed to generate Sub-Tasks. Aborting workflow.")
                return # St
        except Exception as e:
            logger.error(f"[{self.name}] Main agent error: {e}")
            raise
    
    async def _step2_parallel_search_sync(self, ctx: InvocationContext) -> List[Dict[str, str]]:
        
        def _create_branch_ctx_for_sub_agent(
                agent: BaseAgent,
                sub_agent: BaseAgent,
                invocation_context: InvocationContext,
            ) -> InvocationContext:
            """Create isolated branch for every sub-agent."""
            invocation_context = invocation_context.model_copy()
            branch_suffix = f"{agent.name}.{sub_agent.name}"
            invocation_context.branch = (
                f"{invocation_context.branch}.{branch_suffix}"
                if invocation_context.branch
                else branch_suffix
            )
            return invocation_context