import os 
import time 
import threading 

from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv

load_dotenv()

gemini_api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY4"),
]

class GeminiApiKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_count = {key: 0 for key in self.api_keys}

    def get_next_key(self):
        """Lấy key tiếp theo theo round-robin"""
        with self.lock:
            key = self.api_keys[self.current_index]
            self.usage_count[key] += 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            print(f"🔄 Using API Key #{self.current_index} (used {self.usage_count[key]} times)")
            return key
        
    def get_key_for_agent(self, agent_id: int):
        """Lấy key cố định cho agent dựa trên ID (load balancing)"""
        key_index = agent_id % len(self.api_keys)
        key = self.api_keys[key_index]
        self.usage_count[key] += 1
        
        print(f"🎯 Agent {agent_id} -> API Key #{key_index + 1} (used {self.usage_count[key]} times)")
        return key
    
    def print_usage_stats(self):
        """In thống kê sử dụng keys"""
        print("📊 API Key Usage Statistics:")
        total_requests = sum(self.usage_count.values())
        for i, key in enumerate(self.api_keys):
            count = self.usage_count[key]
            percentage = (count / total_requests * 100) if total_requests > 0 else 0
            print(f"  Key #{i+1}: {count} requests ({percentage:.1f}%)")
        print(f"  Total: {total_requests} requests across {len(self.api_keys)} keys")
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_count = {key: 0 for key in self.api_keys}
        print("🔄 API Key usage statistics reset!")


key_manager = GeminiApiKeyManager(gemini_api_keys)

def create_lite_llm_model(model : str, api_base : str, api_key : str):
    """Tạo mô hình Lite LLM
    Args : 
        model (str): Tên mô hình.
        api_base (str): Địa chỉ API.
        api_key (str): Khóa API.
    Return : 
        LiteLlm: Một thể hiện của mô hình Lite LLM.
    """
    return LiteLlm(
        model=model,
        api_base=api_base,
        api_key=api_key
    )


