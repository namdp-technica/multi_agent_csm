#!/usr/bin/env python3
"""
Test VLM sử dụng đúng cách như trong cosmo_workflow.py
"""

import os
import asyncio
import json
from typing import Dict, Any
import google.genai.types as types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import từ agent để lấy vlm_agents
from agent.agent import vlm_agents
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def get_test_images():
    """Lấy danh sách ảnh test từ tools_results"""
    tools_results_dir = "tools_results"
    if not os.path.exists(tools_results_dir):
        logger.error(f"Directory {tools_results_dir} not found")
        return []
    
    images = []
    for filename in os.listdir(tools_results_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(tools_results_dir, filename)
            images.append({
                "id": filename,
                "path": image_path,
                "description": f"Test image from {filename}",
                "relevance_score": 10.0
            })
    
    return images[:3]  # Chỉ test 3 ảnh đầu

async def prepare_vlm_input_with_image(user_query: str, image: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn bị input cho VLM agent - bao gồm text và ảnh theo Google ADK format
    (Copy từ cosmo_workflow.py)
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
                
            logger.info(f"Image prepared: {image_path} ({len(image_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            vlm_input["image_error"] = f"Cannot read image: {e}"
    else:
        logger.warning(f"Image path not found: {image_path}")
        vlm_input["image_error"] = "Image file not found"
    
    return vlm_input

async def test_single_vlm_agent(agent, image: Dict[str, Any], user_query: str):
    """
    Test một VLM agent với một ảnh (copy logic từ cosmo_workflow.py)
    """
    try:
        # Chuẩn bị input cho VLM agent với ảnh
        vlm_input = await prepare_vlm_input_with_image(user_query, image)
        
        logger.info(f"🧪 Testing {agent.name} with image {image.get('id', 'unknown')}...")
        
        # Tạo content với cả text và ảnh
        content_parts = [types.Part(text=vlm_input["text_input"])]
        
        # Thêm ảnh nếu có
        if "image_part" in vlm_input:
            content_parts.append(vlm_input["image_part"])
            logger.info(f"✅ Added image to VLM input: {image.get('id')}")
        else:
            logger.warning(f"❌ No image data for {image.get('id')}")
            return {"error": "No image data"}
        
        # Tạo content với ảnh
        vlm_content = types.Content(
            role='user',
            parts=content_parts
        )
        
        # Tạo session service cho VLM nếu cần
        vlm_session_service = InMemorySessionService()
        
        # Tạo runner cho VLM agent này
        vlm_runner = Runner(
            agent=agent,
            app_name="CosmoVLMTest",
            session_service=vlm_session_service
        )
        
        # Tạo session riêng cho VLM
        vlm_session_id = f"test_{agent.name}_{image.get('id', 'unknown')}"
        vlm_user_id = "test_user"
        
        # Tạo session trước
        await vlm_session_service.create_session(
            app_name="CosmoVLMTest",
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
        
        # Gọi VLM agent với ảnh - giới hạn events
        vlm_response = ""
        event_count = 0
        max_vlm_events = 10  # Giới hạn cho VLM
        
        async for event in events:
            event_count += 1
            logger.info(f"📥 {agent.name} Event #{event_count}")
            
            if hasattr(event, 'content') and event.content and event.content.parts:
                if event.content.parts[0].text:
                    vlm_response = event.content.parts[0].text
                    logger.info(f"💬 Got response: {vlm_response[:100]}...")
                    break  # Break ngay khi có text response
            
            if event_count >= max_vlm_events:
                logger.warning(f"⚠️ {agent.name}: Reached max VLM events limit")
                break
        
        result = {
            "image_id": image.get("id", "unknown"),
            "image_path": image.get("path", ""),
            "vlm_agent": agent.name,
            "response": vlm_response,
            "relevance_score": image.get("relevance_score", 0.0),
            "success": bool(vlm_response)
        }
        
        logger.info(f"✅ {agent.name} completed: {vlm_response[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error testing {agent.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "image_id": image.get("id", "unknown"),
            "vlm_agent": agent.name,
            "response": f"Lỗi xử lý ảnh: {str(e)}",
            "error": str(e),
            "success": False
        }

async def main():
    """Main test function"""
    print("🚀 Cosmo VLM Test Starting...")
    print("="*80)
    
    # Lấy test images
    test_images = get_test_images()
    if not test_images:
        print("❌ No test images found in tools_results")
        return
    
    print(f"📁 Found {len(test_images)} test images:")
    for img in test_images:
        print(f"   - {img['id']} ({img['path']})")
    
    # Test queries
    test_queries = [
        "温度差荷重の記号について教えてください",
        "Describe what you see in this image",
        "What symbols or text are visible in this image?"
    ]
    
    print(f"🤖 Available VLM Agents: {len(vlm_agents)}")
    for i, agent in enumerate(vlm_agents):
        print(f"   {i+1}. {agent.name}")
    
    # Test với agent đầu tiên và ảnh đầu tiên
    test_agent = vlm_agents[0]
    test_image = test_images[0]
    test_query = test_queries[0]
    
    print(f"\n🧪 Testing Configuration:")
    print(f"   Agent: {test_agent.name}")
    print(f"   Image: {test_image['id']}")
    print(f"   Query: {test_query}")
    print("="*80)
    
    # Run test
    result = await test_single_vlm_agent(test_agent, test_image, test_query)
    
    print("\n📊 TEST RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("="*80)
    
    if result.get("success"):
        print("✅ VLM Test PASSED - Image was successfully processed!")
    else:
        print("❌ VLM Test FAILED - Check the error above")
    
    print("🏁 Cosmo VLM Test Completed!")

if __name__ == "__main__":
    asyncio.run(main())