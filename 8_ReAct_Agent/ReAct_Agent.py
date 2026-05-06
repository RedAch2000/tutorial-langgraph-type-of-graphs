from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


load_dotenv()

# email = Annotated[str, "This has to be a valid email format"]

# print(email.__metadata__)

# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state.|
# Tells us how to merge new data into the current state

# Without a reducer, updates would have replaced the existing value entirely!

# Without a reducer
state = {"messages": ["Hi!"]} 
update = {"messages": ["Nice to meet you!"]}
new_state = {"messages": ["Nice to meet you!"]}

# With a reducer
state = {"messages": ["Hi!"]}
update = {"messages": ["Nice to meet you!"]}
new_state = {"messages": ["Hi!", "Nice to meet you!"]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together."""
    return a + b

@tool
def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers together."""
    return a * b

@tool
def subtract(a: int, b: int):
    """This is a subtraction function that subtracts 2 numbers."""
    return a - b


tools = [add, multiply]

model = ChatOllama(model="qwen3.5:2b").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"




graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "our_agent")


app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please")]}
print_stream(app.stream(inputs, stream_mode="values"))