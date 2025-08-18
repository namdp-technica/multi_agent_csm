import threading
import os
import time
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.genai import types

# Load environment variables
load_dotenv()

class ApiKeyManager:
    """Quản lý xoay vòng API keys để tránh quota limit"""
    
    def __init__(self):
        # Đọc API keys từ environment variables
        self.api_keys = []
        
        # Thử đọc nhiều keys trước
        for i in range(1, 10):  # Hỗ trợ tối đa 9 keys
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                self.api_keys.append(key)
        
        # Nếu không có nhiều keys, thử đọc key duy nhất
        if not self.api_keys:
            single_key = os.getenv('GEMINI_API_KEY')
            if single_key:
                self.api_keys = [single_key]
        
        # Fallback nếu không có key nào
        if not self.api_keys:
            print("⚠️  Warning: No API keys found in environment variables!")
            print("   Please set GEMINI_API_KEY or GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in config.env")
            self.api_keys = []
        
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_count = {key: 0 for key in self.api_keys}
        
        print(f"🔑 Initialized API Key Manager with {len(self.api_keys)} keys")
        if self.api_keys:
            print(f"   Keys loaded from environment variables")
    
    def get_next_key(self):
        """Lấy key tiếp theo theo round-robin"""
        if not self.api_keys:
            raise ValueError("No API keys available. Please check your config.env file.")
            
        with self.lock:
            key = self.api_keys[self.current_index]
            self.usage_count[key] += 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            print(f"🔄 Using API Key #{self.current_index} (used {self.usage_count[key]} times)")
            return key
    
    def get_key_for_agent(self, agent_id: int):
        """Lấy key cố định cho agent dựa trên ID (load balancing)"""
        if not self.api_keys:
            raise ValueError("No API keys available. Please check your config.env file.")
            
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

# Global key manager
key_manager = ApiKeyManager()

# ===============================
# Dynamic API Key Setting
# ===============================
def create_agent_with_api_key_rotation(name: str, model: str, agent_id: int, description: str, 
                                     instruction: str, tools=None, output_key=None, temperature= None):
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
        'generate_content_config': types.GenerateContentConfig(temperature=temperature),
        'before_agent_callback': before_callback,
        'after_agent_callback': after_callback
    }
    
    if tools:
        agent_kwargs['tools'] = tools
    if output_key:
        agent_kwargs['output_key'] = output_key
        
    return LlmAgent(**agent_kwargs)