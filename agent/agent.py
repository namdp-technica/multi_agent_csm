import os
import time
from google.adk.agents import LlmAgent 
from tools.tools import VectorDatabaseTool
import prompt 

os.environ['GEMINI_API_KEY'] = "AIzaSyDIIJn9CACPkootvUV26HKYZiczgtD9fO8"

# Tool setup
vector_tool = VectorDatabaseTool()
tool_search = vector_tool.semantic_search
tool_image_search = vector_tool.image_search

# ===============================
# Main Coordinator Agent
# ===============================
main_agent = LlmAgent(
    name="MainCoordinator",
    model="gemini-2.5-pro",
    description="Điều phối tìm kiếm ảnh và phân tích VLM",
    instruction=prompt.MAIN_AGENT_PROMPT,
    output_key="user_query"
)

# ===============================
# Retriever Agent (Image Search)
# ===============================
agent_search = LlmAgent(
    name='RetrieverAgent',
    model="gemini-2.0-flash",
    description="Tìm kiếm ảnh liên quan đến câu hỏi",
    instruction=prompt.SEARCH_AGENT_PROMPT,
    tools=[tool_image_search],
    output_key="retrieved_images"
)

# ===============================
# VLM Agents (Multiple instances)
# ===============================
def create_vlm_agent(agent_id: int):
    """Tạo VLM agent với callback để theo dõi"""
    def before_callback(callback_context):
        print(f"[{time.time():.3f}] 🔄 VLM Agent {agent_id} started analyzing image...")
    
    def after_callback(callback_context):
        print(f"[{time.time():.3f}] ✅ VLM Agent {agent_id} completed analysis!")
    
    return LlmAgent(
        name=f'VLMAgent{agent_id}',
        model="gemini-2.0-flash",
        description=f"VLM Agent #{agent_id} - Phân tích ảnh và trả lời câu hỏi",
        instruction=prompt.VLM_AGENT_PROMPT,
        before_agent_callback=before_callback,
        after_agent_callback=after_callback,
        output_key=f"vlm_result_{agent_id}"
    )

# Tạo danh sách VLM agents
vlm_agents = [create_vlm_agent(i) for i in range(1, 6)]  # 5 VLM agents

# ===============================
# Final Response Agent
# ===============================
final_response_agent = LlmAgent(
    name="FinalResponseAgent",
    model="gemini-2.5-pro", 
    description="Tổng hợp và đưa ra câu trả lời cuối cùng",
    instruction=prompt.FINAL_RESPONSE_PROMPT
)