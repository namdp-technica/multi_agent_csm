#!/usr/bin/env python3
"""
Main entry point for the Cosmo Agent workflow system.
This file sets up the CosmoWorkflow with main agent and sub-agents,
creates the session service, and provides interaction functions.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Import our new VLM workflow and agents
from workflow.cosmo_workflow import CosmoWorkflow
from agent.agent import main_agent, agent_search, vlm_agents, final_response_agent

# --- Constants ---
APP_NAME = "vlm_cosmo_agent"
USER_ID = "user_001"
SESSION_ID = "session_001"

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Create the VLM CosmoWorkflow instance ---
cosmo_workflow = CosmoWorkflow(
    name="VLMCosmoWorkflow",
    main_agent=main_agent,
    retriever_agent=agent_search,
    vlm_agents=vlm_agents,
    final_response_agent=final_response_agent
)

# --- Initial session state ---
INITIAL_STATE = {
    "user_query": "",
    "retrieved_images": {},
    "vlm_responses": {},
    "workflow_status": "initialized"
}

# --- Setup Session and Runner ---
async def setup_session_and_runner():
    """
    Set up the session service and runner for the VLM Cosmo workflow.
    
    Returns:
        tuple: (session_service, runner)
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=INITIAL_STATE
    )
    logger.info(f"Initial session state: {session.state}")
    
    runner = Runner(
        agent=cosmo_workflow,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    return session_service, runner

# --- Main interaction function ---
async def run_vlm_workflow(user_query: str) -> Dict[str, Any]:
    """
    Run the VLM Cosmo workflow with a user query.
    
    Args:
        user_query: The user's question about images
        
    Returns:
        Dict containing the final response and session state
    """
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    logger.info(f"🚀 Starting VLM Cosmo workflow with query: {user_query}")
    logger.info(f"⏰ Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup session and runner
    session_service, runner = await setup_session_and_runner()
    
    # Get current session
    current_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    if not current_session:
        logger.error("❌ Session not found!")
        return {"error": "Session not found"}
    
    # Update session with user query
    current_session.state["user_query"] = user_query
    current_session.state["workflow_status"] = "running"
    logger.info(f"📝 Updated session with user query: {user_query}")
    
    # Create content for the workflow
    content = types.Content(
        role='user',
        parts=[types.Part(text=user_query)]
    )
    
    # Run the workflow
    events = runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    )
    
    # Process events and capture final response
    final_response = "No final response captured."
    event_count = 0
    vlm_responses = []
    
    try:
        async for event in events:
            event_count += 1
            logger.info(f"📨 Event #{event_count} from [{event.author}]")
            
            # Track VLM agent responses
            if event.author.startswith('VLMAgent') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        vlm_responses.append({
                            'agent': event.author,
                            'response': part.text,
                            'timestamp': time.time()
                        })
            
            # Check if this is a final response
            if event.is_final_response() and event.content and event.content.parts and len(event.content.parts) > 0:
                part_text = event.content.parts[0].text
                if part_text:
                    final_response = part_text
                    logger.info(f"✅ Final response captured from [{event.author}]: {final_response[:100]}...")
                
    except Exception as e:
        logger.error(f"❌ Error during workflow execution: {str(e)}")
        return {"error": f"Workflow execution failed: {str(e)}"}
    
    # Get final session state
    final_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Calculate execution time
    end_time = time.time()
    end_datetime = datetime.now()
    execution_time = end_time - start_time
    
    # Update final session state
    final_session_state = {}
    if final_session:
        final_session.state["workflow_status"] = "completed"
        final_session.state["execution_time"] = execution_time
        final_session.state["start_time"] = start_datetime.isoformat()
        final_session.state["end_time"] = end_datetime.isoformat()
        final_session_state = final_session.state
    
    # Return results
    result = {
        "query": user_query,
        "final_response": final_response,
        "vlm_responses": vlm_responses,
        "events_processed": event_count,
        "session_state": final_session_state,
        "status": "success",
        "execution_time": execution_time,
        "start_time": start_datetime.isoformat(),
        "end_time": end_datetime.isoformat()
    }
    
    logger.info(f"🎉 VLM Workflow completed successfully! Processed {event_count} events.")
    logger.info(f"⏰ End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
    return result

# --- Interactive CLI function ---
async def interactive_mode():
    """
    Run the VLM Cosmo workflow in interactive mode.
    """
    print("🤖 Welcome to VLM Cosmo Agent Interactive Mode!")
    print("💡 Ask me questions about images and I'll help you analyze them.")
    print("🚪 Type 'exit' or 'quit' to leave.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🔍 Your image question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye! Thanks for using VLM Cosmo Agent!")
                break
            
            if not user_input:
                print("❓ Please enter a question or type 'exit' to quit.")
                continue
            
            # Run the workflow
            print(f"\n🔄 Processing your query: {user_input}")
            print("⏳ Please wait while I search for images and analyze them...\n")
            
            result = await run_vlm_workflow(user_input)
            
            # Display results
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                print("=" * 60)
                print("🎯 VLM COSMO AGENT RESPONSE:")
                print("=" * 60)
                print(result["final_response"])
                print("=" * 60)
                
                if result["vlm_responses"]:
                    print("📊 VLM AGENT RESPONSES:")
                    for i, vlm_resp in enumerate(result["vlm_responses"], 1):
                        print(f"  {i}. {vlm_resp['agent']}: {vlm_resp['response'][:100]}...")
                    print("=" * 60)
                
                print(f"📊 Events processed: {result['events_processed']}")
                print(f"⏱️ Execution time: {result['execution_time']:.2f}s")
                print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using VLM Cosmo Agent!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in interactive mode: {str(e)}")

# --- Main entry point ---
async def main():
    """
    Main entry point for the VLM Cosmo Agent system.
    """
    logger.info("🚀 Starting VLM Cosmo Agent System")
    
    # Test query for VLM workflow
    test_query = "Tìm cho tôi những hình ảnh về con mèo và cho biết con mèo trong ảnh có màu gì?"
    
    print(f"🔍 Testing with query: {test_query}")
    print("⏳ Please wait while I search for images and analyze them...\n")
    
    # Run the workflow with test query
    result = await run_vlm_workflow(test_query)
    
    # Display results
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print("=" * 60)
        print("🎯 VLM COSMO AGENT RESPONSE:")
        print("=" * 60)
        print(result["final_response"])
        print("=" * 60)
        
        if result["vlm_responses"]:
            print("📊 VLM AGENT RESPONSES:")
            for i, vlm_resp in enumerate(result["vlm_responses"], 1):
                print(f"  {i}. {vlm_resp['agent']}: {vlm_resp['response'][:100]}...")
            print("=" * 60)
        
        print(f"📊 Events processed: {result['events_processed']}")
        print(f"⏱️ Execution time: {result['execution_time']:.2f}s")
        print(f"⏰ Start time: {datetime.fromisoformat(result['start_time']).strftime('%H:%M:%S')}")
        print(f"🏁 End time: {datetime.fromisoformat(result['end_time']).strftime('%H:%M:%S')}")
        
        print("\n📋 Final Session State:")
        session_state_clean = {k: v for k, v in result["session_state"].items() 
                             if k not in ["workflow_execution_time"]}
        print(json.dumps(session_state_clean, indent=2, ensure_ascii=False))
        print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        print(f"❌ Fatal error: {str(e)}")
