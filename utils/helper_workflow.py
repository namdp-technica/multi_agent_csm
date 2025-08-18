import os
import logging
from typing import List, Dict, Any, Optional
from google.adk.agents.invocation_context import InvocationContext
from pathlib import Path
import yaml
import sys

logger = logging.getLogger(__name__)

def extract_user_input(ctx: InvocationContext) -> str:
    """Lấy input từ context"""
    if hasattr(ctx, "request") and ctx.request and ctx.request.content:
        if ctx.request.content.parts:
            return ctx.request.content.parts[0].text
    return ctx.session.state.get("user_query", "")

def distribute_images_to_agents(
    all_images: List[Dict[str, Any]], vlm_agents: List
) -> List[List[Dict[str, Any]]]:
    """Chia ảnh cho các VLM agent"""
    num_agents = len(vlm_agents)
    num_images = len(all_images)

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
                f"[helpers] {vlm_agents[i].name}: Assigned {len(batch)} images"
            )
        start_idx = end_idx

    return image_batches

async def prepare_vlm_input_with_image(
    user_query: str, image: Dict[str, Any], types
) -> Dict[str, Any]:
    """Chuẩn bị input text + ảnh cho VLM agent"""
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

    image_path = image.get("path")
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                image_mime_type = "image/png"
                vlm_input["image_part"] = types.Part(
                    inline_data=types.Blob(
                        mime_type=image_mime_type, data=image_data
                    )
                )
            logger.info(f"[helpers] Image prepared: {image_path} ({len(image_data)} bytes)")
        except Exception as e:
            logger.error(f"[helpers] Error reading image {image_path}: {e}")
            vlm_input["image_error"] = f"Cannot read image: {e}"
    else:
        logger.warning(f"[helpers] Image path not found: {image_path}")
        vlm_input["image_error"] = "Image file not found"

    return vlm_input

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dict containing configuration values.

    Raises:
        SystemExit: If config file not found or cannot be parsed.
    """
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        sys.exit(1)

