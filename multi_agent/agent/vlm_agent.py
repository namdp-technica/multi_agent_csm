from .load_agent import create_agent_with_api_key_rotation
from ..vlm.load_model import create_lite_llm_model
from .prompt import * 

def create_vlm_agent(agent_id : int, provider : str, **kwargs): 
    """Tạo một VLM agent.

    Args:
        agent_id (int): ID của agent.
        provider (str): Nhà cung cấp dịch vụ (ví dụ: "gemini" hoặc "custom").
        **kwargs: Các tham số bổ sung cho mô hình Lite LLM nếu provider là "custom".
            model (str): Tên mô hình.
            api_base (str): Địa chỉ API.
            api_key (str): Khóa API.
    """
    if provider == "gemini" :
        model = "gemini-2.5-pro"
    else : 
        model = create_lite_llm_model(**kwargs)

    return create_agent_with_api_key_rotation(
        name=f"VLMAgent_{agent_id}",
        model=model,
        agent_id=agent_id,
        description=f"VLM Agent #{agent_id} - Phân tích ảnh và trả lời câu hỏi",
        instruction=VLM_AGENT_PROMPT,
        output_key=f"vlm_result_{agent_id}"
    )