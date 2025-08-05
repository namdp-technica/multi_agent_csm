#!/usr/bin/env python3
"""
Test script để kiểm tra VLM có nhận được ảnh từ folder tools_results không
"""

import os
import base64
import asyncio
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.core import InMemorySessionService, Runner
import google.genai.types as types

# Config giống như trong agent.py
os.environ['OPENAI_API_KEY'] = "sk-fake-key-for-local-model"
api_base_url = "http://85.167.195.195:34751/v1"

def load_image_as_base64(image_path):
    """Load ảnh và convert sang base64"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

def get_available_images():
    """Lấy danh sách ảnh có sẵn trong tools_results"""
    tools_results_dir = "tools_results"
    if not os.path.exists(tools_results_dir):
        print(f"❌ Directory {tools_results_dir} not found")
        return []
    
    images = [f for f in os.listdir(tools_results_dir) if f.endswith('.png')]
    return sorted(images)

async def test_vlm_with_image(image_path, question="Describe what you see in this image"):
    """Test VLM với một ảnh cụ thể"""
    print(f"\n🧪 Testing VLM with image: {image_path}")
    print(f"📝 Question: {question}")
    print("="*80)
    
    # Load ảnh
    base64_image = load_image_as_base64(image_path)
    if not base64_image:
        return
    
    # Tạo VLM agent
    vlm_model = LiteLlm(
        model="hosted_vllm/AIDC-AI/Ovis2-4B", 
        api_base=api_base_url,
        api_key="sk-fake-key-for-local-model"
    )
    
    vlm_agent = LlmAgent(
        name="TestVLMAgent",
        model=vlm_model,
        description="Test VLM Agent - Phân tích ảnh",
        instruction="""
Bạn là VLM Agent. Nhiệm vụ của bạn:

1. Nhận câu hỏi và thông tin ảnh từ input
2. Phân tích ảnh được cung cấp và trả lời câu hỏi

QUY TẮC QUAN TRỌNG:
- CHỈ trả lời nếu nội dung ảnh có liên quan trực tiếp đến câu hỏi
- Nếu ảnh KHÔNG liên quan đến câu hỏi, hãy trả lời: "Tôi không biết"
- Nếu ảnh có liên quan, hãy trả lời ngắn gọn, chính xác dựa trên nội dung ảnh
- Không bịa đặt thông tin không có trong ảnh
- Mô tả chi tiết những gì bạn thấy trong ảnh
"""
    )
    
    # Tạo session và runner
    session_service = InMemorySessionService()
    runner = Runner(vlm_agent, session_service)
    session = await session_service.create_session()
    
    try:
        # Tạo content với text và image
        image_part = types.Part(
            inline_data=types.Blob(
                mime_type="image/png",
                data=base64_image
            )
        )
        
        text_part = types.Part(text=f"""
User query: {question}
Image ID: {os.path.basename(image_path)}
Image description: Test image from tools_results folder

Please analyze this image and answer the user's question. Be detailed about what you see.
""")
        
        content = types.Content(
            parts=[text_part, image_part],
            role="user"
        )
        
        print("📤 Sending request to VLM...")
        print(f"   - Image size: {len(base64_image)} chars (base64)")
        print(f"   - Model: hosted_vllm/AIDC-AI/Ovis2-4B")
        print(f"   - API Base: {api_base_url}")
        
        # Gửi request
        event_count = 0
        async for event in runner.run_async(session, new_message=content):
            event_count += 1
            print(f"\n📥 Event #{event_count}:")
            
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"💬 VLM Response: {part.text}")
                            return part.text
            
            # Giới hạn số events để tránh loop vô tận
            if event_count >= 5:
                print("⚠️  Reached max events limit")
                break
        
        print("❌ No valid response received")
        
    except Exception as e:
        print(f"❌ Error during VLM processing: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 VLM Image Test Starting...")
    print("="*80)
    
    # Lấy danh sách ảnh
    images = get_available_images()
    if not images:
        print("❌ No images found in tools_results directory")
        return
    
    print(f"📁 Found {len(images)} images in tools_results:")
    for i, img in enumerate(images[:5]):  # Show first 5
        print(f"   {i+1}. {img}")
    if len(images) > 5:
        print(f"   ... and {len(images) - 5} more")
    
    # Test với ảnh đầu tiên
    test_image = os.path.join("tools_results", images[0])
    
    # Test cases
    test_cases = [
        "Describe what you see in this image in detail",
        "温度差荷重の記号について教えてください",
        "What symbols or text can you see in this image?",
        "このイメージには何が見えますか？"
    ]
    
    for question in test_cases:
        await test_vlm_with_image(test_image, question)
        print("\n" + "="*80)
    
    print("✅ VLM Image Test Completed!")

if __name__ == "__main__":
    asyncio.run(main())