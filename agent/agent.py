import os
from tools import Api
import prompt
from .load_agent import create_agent_with_api_key_rotation, key_manager
from utils.helper_workflow import load_config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
config = load_config(config_path=CONFIG_PATH)

# Nếu dùng Local VLM 
# os.environ['OPENAI_API_KEY'] = config["local_vlm"]["api_key"]  
# api_base_url = config["local_vlm"]["api_base_url"]

# Tool setup
milvus_tool = Api(output_folder=config["paths"]["tools_results"])
tool_image_search = milvus_tool.image_search  


# ===============================
# Main Coordinator Agent
# ===============================
main_agent = create_agent_with_api_key_rotation(
    name=config["agents"]["main"]["name"],
    model=config["agents"]["main"]["model"],
    temperature=config["agents"]["main"]["temperature"],
    agent_id=config["agents"]["main"]["agent_id"],
    description=config["agents"]["main"]["description"],
    instruction=prompt.MAIN_AGENT_PROMPT,
    output_key=config["agents"]["main"]["output_key"]
)

# ===============================
# Search Agents (Multiple instances)
# ===============================
def create_search_agent(agent_id: int):
    """Tạo Search agent với API key rotation"""
    return create_agent_with_api_key_rotation(
        name=config["agents"]["search"]["name_template"].format(id=agent_id),
        model=config["agents"]["search"]["model"],
        agent_id=agent_id,
        description=config["agents"]["search"]["description_template"].format(id=agent_id),
        instruction=prompt.SEARCH_AGENT_PROMPT,
        tools=[tool_image_search],
        output_key=f"search_result_{agent_id}"
    )

# Tạo danh sách Search agents
search_agents = [create_search_agent(i) for i in range(1, 4)]  # 3 Search agents

# ===============================
# VLM Agents 
# ===============================
def create_vlm_agent(agent_id: int):
    """Tạo VLM agent với API key rotation"""
    return create_agent_with_api_key_rotation(
        name=f'VLMAgent{agent_id}',
        model = config["agents"]["vlm"]["model"],
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
    name=config["agents"]["aggregator"]["name"],
    model=config["agents"]["aggregator"]["model"],
    agent_id=config["agents"]["aggregator"]["agent_id"], 
    description=config["agents"]["aggregator"]["description"],
    instruction=prompt.AGGREGATOR_AGENT_PROMPT
)
# In thống kê API key usage
print("\n" + "="*50)
key_manager.print_usage_stats()
print("="*50)
#example local vlm 
        # model=LiteLlm(
        #     model="hosted_vllm/OpenGVLab/InternVL3-8B", 
        #     api_base=api_base_url,
        #     api_key="sk-fake-key-for-local-model"
        # ),

