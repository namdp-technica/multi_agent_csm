from google.adk.agents import LlmAgent 
from tools.tools import VectorDatabaseTool
import prompt 
tool_search = VectorDatabaseTool().semantic_search

""" Định nghĩa 10 sub agent giống nhau"""
sub_agent = []

for i in range(1,11):
    agent_name = f"agent_search_{i}"
    agent = LlmAgent(
        name = agent_name, 
        model = "gemini-2.0-flash",
        tools = [tool_search],
        description = f"Specialist #{i} with vector DB access",
        instruction = prompt.SEARCH_AGENT_PROMPT,
        output_key = "search_result"
    )
    sub_agent.append(agent)


"""Định nghĩa main agent"""
main_agent = LlmAgent(
    name = "Cooordinator",
    model = "gemini-2.5-pro",
    description = "You are a cooordinator agent that will assign tasks to the sub agents and collect the results to return to the user",
    instruction = prompt.MAIN_AGENT_PROMPT,
    output_key = "evaluation_result"
)
