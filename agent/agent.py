import os
import time
import threading
from google.adk.agents import LlmAgent 
from google.adk.models.lite_llm import LiteLlm

from tools import Api
import prompt

# ===============================
# API Key Rotation Manager
# ===============================
class ApiKeyManager:
    """Quản lý xoay vòng API keys để tránh quota limit"""
    
    def __init__(self):
        self.api_keys = [
            "AIzaSyCXvGEt7fgYfeeTXi6idvcGLYyYRd7obuw",  # Key gốc
            "AIzaSyCnNLp6j8j4OEpPOUo53n34QG2P0QeECS0",  # Key 2
            "AIzaSyBZtVuOsoc4ry4aSmcokG0RPFrrSfGqVzQ",  # Key 3
            "AIzaSyD3Gx75zv_HrzXVO3_ei-N_geZCKcaCcyE"   # Key 4
        ]
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_count = {key: 0 for key in self.api_keys}
        
        print(f"🔑 Initialized API Key Manager with {len(self.api_keys)} keys")
    
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

# Global key manager
key_manager = ApiKeyManager()

# Set default key
os.environ['GEMINI_API_KEY'] = key_manager.api_keys[0]
# Local OpenAI-compatible model configuration
os.environ['OPENAI_API_KEY'] = "sk-fake-key-for-local-model"  # Cần có API key format hợp lệ cho LiteLLM
api_base_url = "http://174.78.228.101:40477/v1"

# Tool setup
milvus_tool = Api(output_folder="tools_results")
tool_image_search = milvus_tool.image_search  

# ===============================
# Dynamic API Key Setting
# ===============================
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

# ===============================
# Main Coordinator Agent
# ===============================
main_agent = create_agent_with_api_key_rotation(
    name="MainCoordinator",
    model="gemini-2.5-pro",
    agent_id=0,
    description="Điều phối tìm kiếm ảnh và phân tích VLM",
    instruction=prompt.MAIN_AGENT_PROMPT,
    output_key="user_query"
)

# ===============================
# Search Agents (Multiple instances)
# ===============================
def create_search_agent(agent_id: int):
    """Tạo Search agent với API key rotation"""
    return create_agent_with_api_key_rotation(
        name=f'SearchAgent{agent_id}',
        model= "gemini-2.0-flash",
        agent_id=agent_id,
        description=f"Search Agent #{agent_id} - Tìm kiếm ảnh",
        instruction=prompt.SEARCH_AGENT_PROMPT,
        tools=[tool_image_search],
        output_key=f"search_result_{agent_id}"
    )

# Tạo danh sách Search agents
search_agents = [create_search_agent(i) for i in range(1, 4)]  # 3 Search agents

# ===============================
# VLM Agents (Multiple instances)
# ===============================
def create_vlm_agent(agent_id: int):
    """Tạo VLM agent với API key rotation"""
    return create_agent_with_api_key_rotation(
        name=f'VLMAgent{agent_id}',
        model = "gemini-2.5-pro",
        # model=LiteLlm(
        #     model="hosted_vllm/OpenGVLab/InternVL3-38B", 
        #     api_base=api_base_url,
        #     api_key="sk-fake-key-for-local-model"
        # ),
        agent_id=agent_id + 10,  # Offset để phân bổ key khác với search agents
        description=f"VLM Agent #{agent_id} - Phân tích ảnh và trả lời câu hỏi",
        instruction=prompt.VLM_AGENT_PROMPT,
        output_key=f"vlm_result_{agent_id}"
    )

# Tạo danh sách VLM agents
vlm_agents = [create_vlm_agent(i) for i in range(1, 6)]  # 5 VLM agents

# ===============================
# Aggregator Agent
# ===============================
aggregator_agent = create_agent_with_api_key_rotation(
    name="AggregatorAgent",
    model="gemini-2.5-pro",
    agent_id=99,  # ID cao để đảm bảo sử dụng key khác
    description="Tổng hợp kết quả từ VLM agents và trả lời cuối cùng",
    instruction=prompt.AGGREGATOR_AGENT_PROMPT
)

# In thống kê API key usage
print("\n" + "="*50)
key_manager.print_usage_stats()
print("="*50)

