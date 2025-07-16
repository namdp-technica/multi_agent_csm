import asyncio 
import time 
import warnings

from google.adk import artifacts 
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.cli.utils import logs 
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import Session
from google.genai import types 

from main_agent import agent 
from dotenv import load_dotenv

load_dotenv(override=True)
warnings.filterwarnings('ignore', category=UserWarning)
logs.log_to_tmp_folder()


async def inspect_agent_session(session_service, session, app_name, user_id):
    print("\n===== SESSION INFO =====")
    print(f"Session ID: {session.id}")
    print(f"App Name: {session.app_name}")
    print(f"User ID: {session.user_id}")
    print(f"Last Update: {session.last_update_time}")
    print(f"Current State (Memory): {session.state}")

    print("\n📜 FULL SESSION EVENT LOG")
    if not session.events:
        print("(No events in this session)")
        return
    for idx, e in enumerate(session.events, 1):
        print("=" * 60)
        print(f"Event #{idx}")
        print(f"🕓 Event Type: {getattr(e, 'type', 'N/A')} | Author: {getattr(e, 'author', 'N/A')}")
        if hasattr(e, 'timestamp'):
            print(f"⏰ Timestamp: {e.timestamp}")
        if hasattr(e, 'content') and e.content:
            try:
                print("💬 Content:", e.content.model_dump(exclude_none=True))
            except Exception:
                print("💬 Content:", e.content)
        else:
            print("💬 Content: (None)")
        if hasattr(e, 'actions') and e.actions:
            if hasattr(e.actions, 'state_delta') and e.actions.state_delta:
                print("🧠 State Change:", e.actions.state_delta)
            if hasattr(e.actions, 'artifacts') and e.actions.artifacts:
                print("📦 Artifacts:", e.actions.artifacts)
        else:
            print("⚡ Actions: (None)")
        print("=" * 60)


async def main(): 
    app_name = "my_app"
    user_id_1 = 'user1'
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner( 
        app_name=app_name,
        agent=agent.root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    session_11 = await session_service.create_session( 
        app_name=app_name, user_id=user_id_1
    )
    
    async def run_prompt(session: Session, new_message: str): 
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=new_message)]
        )
        print('** User says:', content.model_dump(exclude_none=True))
        
        sub_agent_responses = []
        
        async for event in runner.run_async(
            user_id=user_id_1,
            session_id=session.id,
            new_message=content,
        ):
            try:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f'** {event.author}: {part.text}')
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
                                    display_text = response_content[:300] + '...' if len(response_content) > 300 else response_content
                                    print(f'    Response: {display_text}')
                                    if event.author.startswith('Agent'):
                                        sub_agent_responses.append({
                                            'agent': event.author,
                                            'response': response_content,
                                            'timestamp': time.time(),
                                            'type': 'function_response'
                                        })
                else:
                    print(f'** {event.author}: [Event Type: {type(event).__name__}]')
            except Exception as e:
                print(f'** {event.author}: [Error processing event: {str(e)}]')
        
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
    await run_prompt(session_11, 'Tell me about AI in healthcare and machine learning in finance and cyber security')
    
    # Lấy lại session mới nhất để đảm bảo có đầy đủ event
    if session_11 is not None:
        session_11 = await session_service.get_session(
            app_name=app_name,
            user_id=user_id_1,
            session_id=session_11.id
        )
    if session_11 is None:
        print("[Warning] session_11 is None after get_session. Cannot inspect or list artifacts.")
        return
    
    stats = agent.get_sub_agent_stats()
    print(f"\n📈 Final Statistics:")
    print(f"Total sub-agent calls: {stats['total_calls']}")
    print(f"Individual calls: {stats['agent_calls']}")

    print(await artifact_service.list_artifact_keys(
        app_name=app_name, user_id=user_id_1, session_id=session_11.id
    ))

    # 👉 New: Inspect all session events and state changes
    await inspect_agent_session(session_service, session_11, app_name, user_id_1)

    end_time = time.time()
    print('------------------------------------')
    print('End time:', end_time)
    print('Total time:', end_time - start_time)


if __name__ == '__main__':
    asyncio.run(main())
