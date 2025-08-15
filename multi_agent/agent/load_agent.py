import os 
import time 
from ..vlm.load_model import key_manager, create_lite_llm_model
from google.adk.agents import LlmAgent

def create_agent_with_api_key_rotation(name: str, model: str, agent_id: int, description: str, 
                                     instruction: str, tools=None, output_key=None):
    """Tạo agent với API key rotation qua callbacks"""
    
    assigned_key = key_manager.get_key_for_agent(agent_id)
    
    def before_callback(callback_context):
        """Set API key trước khi agent chạy"""
        os.environ['GEMINI_API_KEY'] = assigned_key
        print(f"[{time.time():.3f}] 🔑 {name} using API Key #{(agent_id % len(key_manager.api_keys)) + 1}")
    
    def after_callback(callback_context):
        """Callback sau khi agent hoàn thành"""
        print(f"[{time.time():.3f}] ✅ {name} completed!")
    
    agent_kwargs = {
        'name': name,
        'model': model,
        'description': description,
        'instruction': instruction,
        'before_agent_callback': before_callback,
        'after_agent_callback': after_callback
    }
    
    if tools:
        agent_kwargs['tools'] = tools
    if output_key:
        agent_kwargs['output_key'] = output_key
        
    return LlmAgent(**agent_kwargs)
