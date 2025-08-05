#!/usr/bin/env python3
"""
Test 1 VLM agent với 1 ảnh để kiểm tra response
"""

import asyncio
import os
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from agent.agent import vlm_agents

async def test_single_vlm():
    print("🧪 Testing single VLM agent with one image...")
    
    # Chọn 1 ảnh có sẵn
    image_path = "tools_results/doc_27.png"  # Ảnh có score cao nhất từ log trước
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📷 Using image: {image_path}")
    
    # Setup session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="VLMTest", 
        user_id="test_user", 
        session_id="test_session", 
        state={}
    )
    
    # Đọc ảnh
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    print(f"📊 Image size: {len(image_data)} bytes")
    
    # Tạo content với text + ảnh
    user_text = "温度差荷重の記号について教えてください"
    
    content_parts = [
        types.Part(text=user_text),
        types.Part(inline_data=types.Blob(mime_type="image/png", data=image_data))
    ]
    
    content = types.Content(role='user', parts=content_parts)
    
    # Setup runner  
    vlm_agent = vlm_agents[0]  # Sử dụng VLMAgent1
    runner = Runner(agent=vlm_agent, app_name="VLMTest", session_service=session_service)
    
    print(f"🚀 Testing {vlm_agent.name}...")
    
    try:
        events = runner.run_async(
            user_id="test_user",
            session_id="test_session", 
            new_message=content
        )
        
        event_count = 0
        async for event in events:
            event_count += 1
            print(f"📡 Event #{event_count}:")
            
            if hasattr(event, 'content') and event.content and event.content.parts:
                response = event.content.parts[0].text
                print(f"💬 VLM Response: {response}")
                if response and "不知道" not in response and "không biết" not in response:
                    print("✅ VLM nhận được ảnh và trả lời có nội dung!")
                else:
                    print("⚠️ VLM vẫn trả lời 'không biết'")
                break
            
            if event_count >= 3:
                print("⏰ Reached max events")
                break
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_single_vlm())