from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# defining the state
class AgentState(TypedDict):
    messages: List[HumanMessage]

# llm model (in my case i am using local models for now, but you can use open ai model or others)
llm = ChatOllama(
    model="qwen3.5:2b",        # or mistral, phi, etc.
    # temperature=0.9,
    # base_url="http://localhost:11434"  # optional (default)
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


# defining the graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


# main execution
user_input = input("Enter your message: ")
initial_state = AgentState(messages=[HumanMessage(content=user_input)])
agent.invoke(initial_state)