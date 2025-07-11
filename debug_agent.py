#!/usr/bin/env python3
"""
Debug script để xem chi tiết response của sub-agent
"""

import asyncio
import time
import warnings
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.cli.utils import logs
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
import agent

# Load environment variables
load_dotenv(override=True)
warnings.filterwarnings('ignore', category=UserWarning)
logs.log_to_tmp_folder()

class SubAgentResponseTracker:
    """Track and analyze sub-agent responses"""
    
    def __init__(self):
        self.responses = []
        self.function_calls = []
        self.function_responses = []
        
    def add_response(self, author, content, response_type="text"):
        """Add a response to the tracker"""
        self.responses.append({
            'author': author,
            'content': content,
            'type': response_type,
            'timestamp': time.time()
        })
        
    def add_function_call(self, author, function_name, args):
        """Add a function call to the tracker"""
        self.function_calls.append({
            'author': author,
            'function': function_name,
            'args': args,
            'timestamp': time.time()
        })
        
    def add_function_response(self, author, function_name, response_content):
        """Add a function response to the tracker"""
        self.function_responses.append({
            'author': author,
            'function': function_name,
            'response': response_content,
            'timestamp': time.time()
        })
        
    def print_summary(self):
        """Print a detailed summary of all tracked events"""
        print("\n" + "="*80)
        print("🔍 DETAILED SUB-AGENT ANALYSIS")
        print("="*80)
        
        # Function calls
        if self.function_calls:
            print(f"\n📞 FUNCTION CALLS ({len(self.function_calls)}):")
            for i, call in enumerate(self.function_calls, 1):
                print(f"  {i}. {call['author']} -> {call['function']}")
                print(f"     Args: {call['args']}")
                
        # Function responses
        if self.function_responses:
            print(f"\n📤 FUNCTION RESPONSES ({len(self.function_responses)}):")
            for i, resp in enumerate(self.function_responses, 1):
                print(f"  {i}. {resp['author']} <- {resp['function']}")
                content = resp['response']
                if len(content) > 200:
                    print(f"     Response: {content[:200]}...")
                else:
                    print(f"     Response: {content}")
                    
        # Text responses
        if self.responses:
            print(f"\n💬 TEXT RESPONSES ({len(self.responses)}):")
            for i, resp in enumerate(self.responses, 1):
                print(f"  {i}. {resp['author']}:")
                content = resp['content']
                if len(content) > 200:
                    print(f"     {content[:200]}...")
                else:
                    print(f"     {content}")
                    
        print("="*80)

async def debug_sub_agents():
    """Debug sub-agent responses in detail"""
    
    app_name = "debug_app"
    user_id = 'debug_user'
    
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    
    # Reset counter for clean start
    agent.reset_sub_agent_counter()
    
    runner = Runner(
        app_name=app_name,
        agent=agent.main_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )
    
    # Initialize response tracker
    tracker = SubAgentResponseTracker()
    
    async def run_debug_prompt(session: Session, message: str):
        content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=message)]
        )
        
        print(f'🔍 Debug Query: {message}')
        print('-' * 50)
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            try:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f'📝 {event.author}: {part.text[:100]}...')
                            tracker.add_response(event.author, part.text, "text")
                            
                        elif hasattr(part, 'function_call') and part.function_call:
                            print(f'🔧 {event.author}: [CALL] {part.function_call.name}')
                            args = part.function_call.args if hasattr(part.function_call, 'args') else {}
                            tracker.add_function_call(event.author, part.function_call.name, args)
                            
                        elif hasattr(part, 'function_response') and part.function_response:
                            print(f'📤 {event.author}: [RESPONSE] {part.function_response.name}')
                            response_content = ""
                            if hasattr(part.function_response, 'response') and part.function_response.response:
                                response_content = part.function_response.response.get('content', '')
                            tracker.add_function_response(event.author, part.function_response.name, response_content)
                            
                else:
                    print(f'⚡ {event.author}: [EVENT] {type(event).__name__}')
                    
            except Exception as e:
                print(f'❌ Error processing event: {str(e)}')
    
    # Run debug with a complex query
    await run_debug_prompt(session, 'Explain AI applications in healthcare and machine learning in finance with specific examples')
    
    # Print detailed analysis
    tracker.print_summary()
    
    # Print final statistics
    stats = agent.get_sub_agent_stats()
    print(f"\n📊 FINAL STATISTICS:")
    print(f"Total sub-agent calls: {stats['total_calls']}")
    for agent_name, calls in stats['agent_calls'].items():
        if calls > 0:
            print(f"  {agent_name}: {calls} calls")

if __name__ == '__main__':
    asyncio.run(debug_sub_agents()) 