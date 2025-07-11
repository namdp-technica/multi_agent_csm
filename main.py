import asyncio 
import time 
import warnings

from google.adk import artifacts 

import agent 
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.cli.utils import logs 
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import Session
from google.genai import types 


load_dotenv(override = True)
warnings.filterwarnings('ignore', category = UserWarning)
logs.log_to_tmp_folder()


async def main(): 
    app_name = "my_app"
    user_id_1 = 'user1'
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner( 
        app_name= app_name,
        agent = agent.main_agent,
        artifact_service = artifact_service,
        session_service = session_service,
    )
    session_11 = await session_service.create_session( 
        app_name= app_name, user_id = user_id_1
    )
    
    async def run_prompt(session: Session, new_message: str): 
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=new_message)]
        )
        print('** User says:', content.model_dump(exclude_none=True))
        
        # Track sub-agent responses
        sub_agent_responses = []
        
        async for event in runner.run_async(
            user_id=user_id_1,
            session_id=session.id,
            new_message=content,
        ):
            try:
                if event.content and event.content.parts:
                    # Handle different types of content parts
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f'** {event.author}: {part.text}')
                            
                            # Track sub-agent responses
                            if event.author.startswith('Agent'):
                                sub_agent_responses.append({
                                    'agent': event.author,
                                    'response': part.text,
                                    'timestamp': time.time()
                                })
                                
                        elif hasattr(part, 'function_call') and part.function_call:
                            print(f'** {event.author}: [Function Call] {part.function_call.name}')
                            if hasattr(part.function_call, 'args') and part.function_call.args:
                                print(f'    Args: {part.function_call.args}')
                                
                        elif hasattr(part, 'function_response') and part.function_response:
                            print(f'** {event.author}: [Function Response] {part.function_response.name}')
                            if hasattr(part.function_response, 'response') and part.function_response.response:
                                response_content = part.function_response.response.get('content', '')
                                if response_content:
                                    # Truncate long responses for display
                                    display_text = response_content[:300] + '...' if len(response_content) > 300 else response_content
                                    print(f'    Response: {display_text}')
                                    
                                    # Store full response for sub-agents
                                    if event.author.startswith('Agent'):
                                        sub_agent_responses.append({
                                            'agent': event.author,
                                            'response': response_content,
                                            'timestamp': time.time(),
                                            'type': 'function_response'
                                        })
                                        
                else:
                    # Handle events without content (like function calls)
                    print(f'** {event.author}: [Event Type: {type(event).__name__}]')
                    
            except Exception as e:
                print(f'** {event.author}: [Error processing event: {str(e)}]')
        
        # Print summary of sub-agent responses
        if sub_agent_responses:
            print("\n" + "="*60)
            print("📊 SUB-AGENT RESPONSES SUMMARY")
            print("="*60)
            for i, response in enumerate(sub_agent_responses, 1):
                print(f"\n{i}. {response['agent']}:")
                if response.get('type') == 'function_response':
                    print(f"   [Function Response] {response['response'][:200]}...")
                else:
                    print(f"   {response['response'][:200]}...")
            print("="*60)

    start_time = time.time()
    print('Start time:', start_time)
    print('------------------------------------')
    await run_prompt(session_11, 'Tell me about AI in healthcare and machine learning in finance')
    
    # Get final statistics
    stats = agent.get_sub_agent_stats()
    print(f"\n📈 Final Statistics:")
    print(f"Total sub-agent calls: {stats['total_calls']}")
    print(f"Individual calls: {stats['agent_calls']}")

    print(
        await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id_1, session_id=session_11.id
        )
    )
    end_time = time.time()
    print('------------------------------------')
    print('End time:', end_time)
    print('Total time:', end_time - start_time)


if __name__ == '__main__':
    asyncio.run(main())