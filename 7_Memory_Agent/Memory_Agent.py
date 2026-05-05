from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# defining the state
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
  

# llm model (in my case i am using local models for now, but you can use open ai model or others)
llm = ChatOllama(
    model="qwen3.5:2b",        # or mistral, phi, etc.
    # temperature=0.9,
    # base_url="http://localhost:11434"  # optional (default)
)


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    state["messages"].append(AIMessage(content=response.content))

    return state

# Creating the agent
graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# Testing the agent
conversation_history = []
user_input = input("You (User): ")

while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]  
    user_input = input("You (User): ")

