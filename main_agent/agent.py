import os
import asyncio
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.genai import types
from tool import VectorDatabaseTool
from custom_workflow import CustomSelectiveAgent10  # ✅ Custom Agent
import time

os.environ['GEMINI_API_KEY'] = "AIzaSyDIIJn9CACPkootvUV26HKYZiczgtD9fO8"

# ===============================
# Tool setup
# ===============================
tool_search = VectorDatabaseTool().semantic_search

# Sub-agent usage counter (optional)
sub_agent_call_counter = {
    'total_calls': 0,
    'agent_calls': {f'Agent{i}': 0 for i in range(1, 11)}
}

def get_sub_agent_stats():
    """Get current sub-agent usage statistics"""
    return sub_agent_call_counter.copy()

# ===============================
# Define 10 sub-agents
# ===============================
tool_agents = []
for i in range(1, 11):
    agent_name = f"Agent{i}"

    def create_before_callback(agent_id):
        def before_callback(callback_context):
            sub_agent_call_counter['total_calls'] += 1
            sub_agent_call_counter['agent_calls'][f'Agent{agent_id}'] += 1
            print(f"[{time.time():.3f}] 🔄 Sub-Agent {agent_id} called! Total: {sub_agent_call_counter['total_calls']}")
        return before_callback

    def create_after_callback(agent_id):
        def after_callback(callback_context):
            print(f"[{time.time():.3f}] ✅ Sub-Agent {agent_id} completed!")
        return after_callback

    agent = LlmAgent(
        name=agent_name,
        model="gemini-2.0-flash",
        description=f"Specialist #{i} with vector DB access",
        instruction="""
Bạn là sub-agent. Chỉ thực hiện đúng 1 truy vấn được giao trong input. Không tự động chia nhỏ, không tự động mở rộng, không trả lời các chủ đề khác. Trả lời ngắn gọn, chỉ dựa trên kết quả truy vấn vector database.
1. Sử dụng semantic search tool với input được giao.
2. Trả về thông tin liên quan nhất, trích dẫn nguồn doc_[id].
""",
        tools=[tool_search],
        before_agent_callback=create_before_callback(i),
        after_agent_callback=create_after_callback(i)
    )
    tool_agents.append(agent)

# ===============================
# Main agent
# ===============================
main_agent = LlmAgent(
    name="MainCoordinator",
    model="gemini-2.5-pro",
    description="Decomposes user query into sub-agent tasks",
    instruction="""
## Task
Decompose the user query into sub-queries, and return a JSON list of sub-agents to activate.

**Output only raw JSON. Do NOT use code blocks, markdown, or any explanation. Only output the JSON array.**

- Use only these agent names: Agent1, Agent2, Agent3, Agent4, Agent5, Agent6, Agent7, Agent8, Agent9, Agent10.
- Assign each sub-query to one of the above agents.

Example:
[
  {"agent": "Agent1", "query": "AI in healthcare"},
  {"agent": "Agent3", "query": "AI in finance"}
]
""",
    output_key="sub_agent_plan"
)

# ===============================
# Aggregator agent
# ===============================
aggregator_agent = LlmAgent(
    name="Aggregator",
    model="gemini-2.5-pro",
    description="Aggregates outputs from sub-agents",
    instruction="""
You are given outputs from multiple sub-agents in {sub_agent_outputs}.
Summarize the information, group by topic, and cite sources.
If no info is present, say: "No relevant info found."
"""
)

# ===============================
# Workflow: main -> selective -> aggregator
# ===============================
root_agent = SequentialAgent(
    name="MainWorkflow",
    sub_agents=[
        main_agent,
        CustomSelectiveAgent10(
            name="SelectiveSubAgentRunner",
            sub_agents = tool_agents,
            output_key="sub_agent_outputs"
        ),
        aggregator_agent]
)