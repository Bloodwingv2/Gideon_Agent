# Import Async libraries for co-routines
import asyncio

# Import Annotation libraries for the model and developers
from typing import TypedDict, Annotated, Sequence

# import Multiple MCP client libraries and chat model initialization
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

client = MultiServerMCPClient(
    {
        "Chrome-MCP-Server": {
            "url": "http://127.0.0.1:12306/mcp",
            "transport": "streamable_http",
        }
    }
)

# Move these inside an async function since get_tools() is async
async def setup_agent():
    mcp_tools = await client.get_tools() 
    Ollama_model = init_chat_model("ollama:llama3.2:latest", streaming = True)  # Initialize our Ollama model
    ollama_mcp = Ollama_model.bind_tools(tools=mcp_tools)  # Bind MCP tools to our Model
    return ollama_mcp, mcp_tools

# Define Agent State Class for Gideon Agent, although we are currently only gonna use 1 string
class Gideon(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]

async def model_call(state: Gideon) -> Gideon:
    """Append the System Message and User message to the state"""
    System_message = SystemMessage(content="Your Name is Gideon, You were made by Mirang Bhandari, you have access to various MCP tools via Chrome MCP server which can open webpages, Take screenshots, Add Bookmarks, delete Bookmarks and a lot more, use these tools when necessary otherwise answer the question")
    messages = [System_message] + state["messages"]
    
    # Actually call the model to get a response via thread method
    response = await ollama_mcp.ainvoke(messages)
    
    return {"messages": [response]}  # Return the response message and store it in the state

# Define a Schema where the flow should continue and execute remaining tool calls or end
def should_continue(state: Gideon):
    """Function to control the flow of our Conditional edge"""
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "Continue"
    else:
        return "END"

# Start Main Function code
async def main():
    global ollama_mcp, mcp_tools
    
    # Setup the agent first
    ollama_mcp, mcp_tools = await setup_agent()
    
    # Define the graph flow and conditional edges
    graph = StateGraph(Gideon)
    graph.add_node("Gideon", model_call)
    tool_node = ToolNode(tools=mcp_tools)
    graph.add_node("Tools", tool_node)

    graph.set_entry_point("Gideon")
    graph.add_conditional_edges(
        "Gideon",
        should_continue,
        {
            "Continue": "Tools",
            "END": END
        },
    )
    graph.add_edge("Tools", "Gideon")

    app = graph.compile()
    
    # Main User Loop Begins
    prompt = ""
    print("Hello! my name is Gideon!, How may i assist you today?")
    while "exit" not in prompt.lower():
        prompt = input("User: ")
        if "exit" in prompt.lower():  # Check exit before processing
            break
            
        input_state = {"messages": [HumanMessage(content=prompt)]}  # Use actual prompt
        
        events = app.astream_events(input=input_state, version="v2")
        print("Gideon: ", end="", flush=True)
        async for event in events:
            if event['event'] == "on_chat_model_stream":
                print(event["data"]["chunk"].content, end="",flush = True)
        print("\n")

    print("Exiting System.... Have a Great day!")

# Run the main function
if __name__ =="__main__":
    asyncio.run(main())