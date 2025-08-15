from .load_agent import create_agent_with_api_key_rotation
from ..vlm.load_model import create_lite_llm_model
from .prompt import * 

def create_main_agent(provider : str, **kwargs):
    """Tạo Main agent.
    Args : 
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
        name="MainCoordinator",
        model=model,
        agent_id=0,
        description="Điều phối tìm kiếm ảnh và phân tích VLM",
        instruction=MAIN_AGENT_PROMPT,
        output_key="user_query"
    )
