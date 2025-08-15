from .aggregator_agent import create_aggregator_agent
from .main_agent import create_main_agent
from .search_agent import create_search_agent
from .vlm_agent import create_vlm_agent

aggregator_agent = create_aggregator_agent(provider="gemini")
main_agent = create_main_agent(provider="gemini")

search_agents = [create_search_agent(i, provider="gemini") for i in range(1,4)]
vlm_agents = [create_vlm_agent(i, provider="gemini") for i in range(5,10)]