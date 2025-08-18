import asyncio
import logging
from google.genai import types
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from utils.helper_workflow import distribute_images_to_agents, prepare_vlm_input_with_image

logger = logging.getLogger(__name__)

async def run_vlm_agent_batch(agent, images_batch, user_input, agent_idx):
    batch_results = []

    for i, image in enumerate(images_batch):
        try:
            vlm_input = await prepare_vlm_input_with_image(user_input, image, types)

            content_parts = [types.Part(text=vlm_input["text_input"])]
            if "image_part" in vlm_input:
                content_parts.append(vlm_input["image_part"])

            vlm_content = types.Content(role="user", parts=content_parts)
            vlm_session_service = InMemorySessionService()
            vlm_runner = Runner(
                agent=agent,
                app_name="CosmoVLM",
                session_service=vlm_session_service,
            )

            vlm_session_id = f"vlm_{agent.name}_{image.get('id', 'unknown')}"
            vlm_user_id = "vlm_user"
            await vlm_session_service.create_session(
                app_name="CosmoVLM",
                user_id=vlm_user_id,
                session_id=vlm_session_id,
                state={},
            )

            vlm_response = ""
            async for event in vlm_runner.run_async(
                user_id=vlm_user_id, session_id=vlm_session_id, new_message=vlm_content
            ):
                if event.content and event.content.parts and event.content.parts[0].text:
                    vlm_response = event.content.parts[0].text
                    break

            batch_results.append(
                {
                    "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                    "vlm_agent": agent.name,
                    "response": vlm_response,
                }
            )
            logger.info(f"[vlm_runner] {agent.name} processed {image.get('id')}")
        except Exception as e:
            logger.error(f"[vlm_runner] Error: {e}")
            batch_results.append(
                {
                    "image_id": image.get("id", f"batch_{agent_idx}_img_{i}"),
                    "vlm_agent": agent.name,
                    "response": f"Lỗi xử lý ảnh: {str(e)}",
                }
            )
    return batch_results

async def run_all_vlm_batches(user_input, search_results, vlm_agents):
    """Chạy song song tất cả VLM agents"""
    image_batches = distribute_images_to_agents(search_results, vlm_agents)
    tasks = [
        run_vlm_agent_batch(vlm_agents[i], batch, user_input, i + 1)
        for i, batch in enumerate(image_batches) if batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for r in results:
        if isinstance(r, list):
            all_results.extend(r)
    return all_results
