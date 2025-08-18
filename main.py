import os
import asyncio
import logging
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from workflow.cosmo_workflow import CosmoFlowAgent
from agent.agent import main_agent, search_agents, vlm_agents, aggregator_agent
from utils.helper_workflow import load_config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
config = load_config(config_path=CONFIG_PATH)
load_dotenv()

# --- Constants ---
APP_NAME = config["app"]["name"]
USER_ID = config["app"]["user_id"]
SESSION_ID = config["app"]["session_id"]

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cosmo_flow_agent = CosmoFlowAgent(  
    name="CosmoFlowAgent",  
    main_agent=main_agent,  
    search_agents=search_agents,  
    vlm_agents=vlm_agents,  
    aggregator_agent=aggregator_agent,  
)

async def setup_session_and_runner():
    """Setup session service và runner"""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        state={}
    )
    logger.info(f"Session created: {session.state}")
    
    runner = Runner(
        agent=cosmo_flow_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    return session_service, runner


async def run_cosmo_workflow(user_query: str):
    """
    Chạy Cosmo workflow với user query
    
    Args:
        user_query (str): Câu hỏi từ người dùng
        
    Returns:
        str: Câu trả lời cuối cùng
    """
    logger.info(f"🚀 Starting Cosmo Workflow")
    logger.info(f"📝 User Query: {user_query}")
    logger.info("="*80)
    
    try:
        # Setup session và runner
        session_service, runner = await setup_session_and_runner()
        
        # Tạo content từ user query
        content = types.Content(
            role='user', 
            parts=[types.Part(text=user_query)]
        )
        
        # Chạy workflow
        logger.info("🎯 Executing workflow...")
        events = runner.run_async(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        )
        
        # Thu thập kết quả
        final_response = "No final response captured."
        async for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                logger.info(f"📋 Final response from [{event.author}]")
                final_response = event.content.parts[0].text
        
        # In kết quả
        print("\n" + "="*80)
        print("🎉 COSMO WORKFLOW RESULT")
        print("="*80)
        print(f"📝 Query: {user_query}")
        print(f"💬 Answer: {final_response}")
        print("="*80)
        
        # In thống kê session
        final_session = await session_service.get_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=SESSION_ID
        )
        
        if final_session:
            logger.info("📊 Final Session State:")
            for key, value in final_session.state.items():
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value[:100]}...")
                else:
                    logger.info(f"  {key}: {value}")
        
        return final_response
        
    except Exception as e:
        logger.error(f"❌ Workflow error: {str(e)}")
        error_message = f"Xin lỗi, có lỗi xảy ra trong quá trình xử lý: {str(e)}"
        print(f"\n❌ Error: {error_message}")
        return error_message


def main():
    """Main function để chạy từ command line"""
    print("🌟 Welcome to COSMO - Visual Question Answering System")
    print("="*60)
    
    # Test queries
    test_queries = [
        "温度差荷重の記号について教えてください",
        "Explain the symbols for temperature differential loads",
        "Cấu trúc của một tòa nhà như thế nào?",
    ]
    
    print("🧪 Available test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "="*60)
    
    # Cho phép user nhập query hoặc chọn test query
    user_input = input("Enter your query (or press Enter for test query 1): ").strip()
    
    # Xử lý lựa chọn số
    if user_input.isdigit():
        choice = int(user_input) - 1
        if 0 <= choice < len(test_queries):
            user_input = test_queries[choice]
            print(f"Using test query {choice + 1}: {user_input}")
        else:
            user_input = test_queries[0]
            print(f"Invalid choice, using test query 1: {user_input}")
    elif not user_input:
        user_input = test_queries[0]
        print(f"Using test query 1: {user_input}")
    
    # Chạy workflow
    try:
        result = asyncio.run(run_cosmo_workflow(user_input))
        print(f"\n✅ Workflow completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Workflow interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")


if __name__ == "__main__":
    main()
