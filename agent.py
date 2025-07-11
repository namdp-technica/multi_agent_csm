import os
import asyncio
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools import agent_tool
from google.adk.runners import Runner
from google.genai import types

from tool import VectorDatabaseTool
os.environ['GEMINI_API_KEY'] = "AIzaSyDIIJn9CACPkootvUV26HKYZiczgtD9fO8"

# Initialize vector database tool
tool_search = VectorDatabaseTool().semantic_search

# Global counter for sub-agent calls
sub_agent_call_counter = {
    'total_calls': 0,
    'agent_calls': {f'Agent{i}': 0 for i in range(1, 11)}
}

# Tạo 10 chuyên gia dưới dạng ToolAgent với vector database access
tool_agents = []
for i in range(1,11):
    agent_name = f"Agent{i}"
    
    # Create callback functions for this specific agent
    def create_before_callback(agent_id):
        def before_callback(callback_context):
            sub_agent_call_counter['total_calls'] += 1
            sub_agent_call_counter['agent_calls'][f'Agent{agent_id}'] += 1
            print(f'🔄 Sub-Agent {agent_id} called! (Total calls: {sub_agent_call_counter["total_calls"]})')
        return before_callback
    
    def create_after_callback(agent_id):
        def after_callback(callback_context):
            print(f'✅ Sub-Agent {agent_id} completed!')
        return after_callback
    
    sub = LlmAgent(
        name=agent_name,
        model="gemini-2.0-flash",
        description=f"Chuyên trách nhiệm vụ #{i} với khả năng truy vấn vector database",
        instruction="""
Each Sub-Agent is responsible for executing a specific retrieval task against a vector database. You operate under the direction of the Main Agent and use a **Retrieval Tool** to fetch relevant documents or embeddings. Your goal is to return accurate, concise, and relevant results for the assigned sub-query.

You should:
1. Use the semantic search tool to find relevant documents
2. Return only the most relevant information found
3. Cite sources using doc_[document_id] format
4. Be concise and focused on the specific sub-query assigned
        """,
        tools=[tool_search],  # Add vector database tool to each agent
        before_agent_callback=create_before_callback(i),
        after_agent_callback=create_after_callback(i)
    )
    tool_agents.append(agent_tool.AgentTool(agent=sub))

# Essential callback functions
async def before_agent_callback(callback_context):
    print('🔄 Main Agent started')
    return None

async def after_agent_callback(callback_context):
    print('✅ Main Agent completed')
    return None

async def before_model_callback(callback_context, llm_request):
    print('🤖 Model request initiated')
    return None

async def after_model_callback(callback_context, llm_response):
    print('🤖 Model response received')
    return None

def print_sub_agent_stats(callback_context):
    """Print final statistics of sub-agent usage"""
    print("\n" + "="*50)
    print("📊 SUB-AGENT USAGE STATISTICS")
    print("="*50)
    print(f"Total sub-agent calls: {sub_agent_call_counter['total_calls']}")
    print("\nIndividual agent calls:")
    for agent_name, calls in sub_agent_call_counter['agent_calls'].items():
        if calls > 0:
            print(f"  {agent_name}: {calls} calls")
    print("="*50)

def get_sub_agent_stats():
    """Get current sub-agent usage statistics"""
    return sub_agent_call_counter.copy()

def reset_sub_agent_counter():
    """Reset the sub-agent call counter"""
    global sub_agent_call_counter
    sub_agent_call_counter = {
        'total_calls': 0,
        'agent_calls': {f'Agent{i}': 0 for i in range(1, 11)}
    }
    print("🔄 Sub-agent counter reset!")

main_agent = LlmAgent(
    name="MainCoordinator",
    model="gemini-2.5-pro",
    description="Điều phối nhiệm vụ & phân bổ cho các sub-agent",
    instruction="""
## Role

You are the Main Agent in a multi-agent AI system. Your core responsibility is to:
- Receive a user query
- Decompose the query into smaller sub-tasks if needed
- Assign these sub-tasks to a subset of 10 available Sub-Agents, each capable of querying a vector database
- Aggregate the retrieved results
- Compose a final answer **strictly based on those retrieved results**

## Core Responsibilities

### 1. Understand the User Query
- Analyze the query to identify its intent, structure, and potential subtopics.
- If the query is simple, it can be processed directly without decomposition.

### 2. Task Decomposition
- Break down complex or multi-faceted queries into multiple sub-queries, each addressing a specific aspect of the original query.

### 3. Optimize Sub-Agent Usage
- Choose from 10 sub-agents, all with the same capability: querying from a vector database.
- Decide how many sub-agents to activate based on task complexity, required accuracy, and performance needs.
- **Do not call sub-agents unnecessarily.**

### 4. Aggregate Responses
- **Only** collect and use the **exact** contents returned by sub-agents.
- Remove duplicates or clearly irrelevant entries.
- **Do not invent, infer or extrapolate** beyond what was retrieved.
- For any sub-query that yields no results, note it as "No information found for this sub-topic."

### 5. Respond to the User
- If **at least one** retrieval result exists:
  1. Present a clear, concise summary grouped by sub-topic.
  2. Quote or paraphrase **only** the retrieved text (include citation: `doc_[document_id]`).
- If **no** retrieval results were found for the entire query:
  - Reply:  
    > "I'm sorry, I couldn't find any relevant information in the database for your query."

- Use structured format (e.g., bullet points or sections) when covering multiple aspects.

## Sub-Agent Allocation Strategy

| Situation                          | Recommended Strategy                                |
|------------------------------------|-----------------------------------------------------|
| Simple query                       | Use 1 sub-agent for efficiency                      |
| Multi-aspect query                 | Decompose and assign 1–2 agents per aspect          |
| Verification required              | Assign multiple agents to same sub-task             |
| Fast response needed               | Query multiple agents in parallel                   |

## Example

**User Query:**  
"What are the real‑world applications of transformers in healthcare, finance, and education?"

**Main Agent Behavior:**
1. Recognize 3 domains → decompose into 3 sub‑queries.
2. Assign 1–2 sub‑agents per domain for robust retrieval.
3. Aggregate **only retrieved** results:
   - If `doc_1`, `doc_2`, `doc_3` returned, quote/paraphrase and cite each.
4. Format final answer into 3 sections (Healthcare, Finance, Education).
5. If any section has no docs, explicitly state "No information found for [Domain].""",
    tools=tool_agents,
    before_agent_callback=before_agent_callback,
    after_agent_callback=[after_agent_callback, print_sub_agent_stats],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)

# Ví dụ dùng Sequential + Parallel pattern:
workflow = SequentialAgent(
    name="MainWorkflow",
    sub_agents=[
        main_agent,  # xác định kế hoạch
        ParallelAgent(name="ParallelTools", sub_agents=[t.agent for t in tool_agents] ),
        # (Có thể thêm bước tổng hợp ở đây)
    ]
)