#!/usr/bin/env python3
"""
Test VLM agents với mock data để kiểm tra ảnh có được truyền đúng không
"""

import asyncio
import os
from workflow.cosmo_workflow import cosmo_flow_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# --- Test VLM với mock data ---
async def test_vlm_with_mock_images():
    print("🧪 Testing VLM Agents with mock images...")
    
    # Mock images data - sử dụng ảnh có sẵn trong tools_results
    mock_images = []
    tools_results_dir = "tools_results"
    
    if os.path.exists(tools_results_dir):
        image_files = [f for f in os.listdir(tools_results_dir) if f.endswith('.png')][:3]  # Lấy 3 ảnh đầu
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(tools_results_dir, img_file)
            if os.path.exists(img_path):
                mock_images.append({
                    "id": f"test_{img_file.replace('.png', '')}",
                    "path": img_path,
                    "url": None,
                    "description": f"Test image {img_file}",
                    "relevance_score": 0.9,
                    "metadata": {"format": "png"}
                })
    
    if not mock_images:
        print("❌ No test images found in tools_results/")
        return
    
    print(f"📷 Found {len(mock_images)} test images:")
    for img in mock_images:
        print(f"  - {img['id']}: {img['path']}")
    
    # Test VLM agent đầu tiên với ảnh đầu tiên
    from agent.agent import vlm_agents
    vlm_agent = vlm_agents[0]
    test_image = mock_images[0]
    
    print(f"\n🔍 Testing image preparation for {test_image['id']}...")
    
    # Test method chuẩn bị input với ảnh
    try:
        vlm_input = await cosmo_flow_agent._prepare_vlm_input_with_image("温度差荷重の記号について教えてください", test_image)
        
        print(f"📝 Text input: {vlm_input['text_input'][:100]}...")
        
        if "image_part" in vlm_input:
            print(f"✅ Image part created successfully!")
            print(f"📊 Image part type: {type(vlm_input['image_part'])}")
        else:
            print(f"❌ No image_part in VLM input!")
            if "image_error" in vlm_input:
                print(f"   Error: {vlm_input['image_error']}")
        
        # Test tạo content với ảnh
        content_parts = [types.Part(text=vlm_input["text_input"])]
        
        if "image_part" in vlm_input:
            content_parts.append(vlm_input["image_part"])
            print(f"✅ Content parts created: {len(content_parts)} parts")
            print(f"   - Part 1: {type(content_parts[0])} (text)")
            print(f"   - Part 2: {type(content_parts[1])} (image)")
        
        # Test tạo full content
        vlm_content = types.Content(role='user', parts=content_parts)
        print(f"✅ Content object created with {len(vlm_content.parts)} parts")
        
    except Exception as e:
        print(f"❌ Error preparing VLM input: {str(e)}")
        import traceback
        traceback.print_exc()

# Test function
if __name__ == "__main__":
    asyncio.run(test_vlm_with_mock_images())