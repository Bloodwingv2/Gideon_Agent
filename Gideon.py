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

# Initialize MCP Client
client = MultiServerMCPClient(
    {
        "Chrome-MCP-Server": {
            "url": "http://127.0.0.1:12306/mcp",
            "transport": "streamable_http",
        }
    }
)

# Define Agent State Class for Gideon Agent
class Gideon(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]

async def initialize_agent():
    """Initialize the agent with MCP tools and model"""
    mcp_tools = await client.get_tools()  # Retrieve MCP Tools
    ollama_model = init_chat_model("ollama:llama3.2:latest")  # Initialize our Ollama model
    ollama_mcp = ollama_model.bind_tools(tools=mcp_tools)  # Bind MCP tools to our Model
    return ollama_mcp, mcp_tools

async def model_call(state: Gideon, model) -> Gideon:
    """Process the messages and get model response"""
    system_message = SystemMessage(
        content="Your Name is Gideon, You were made by Mirang Bhandari, you have access to various MCP tools via Chrome MCP server which can open webpages, Take screenshots, Add Bookmarks, delete Bookmarks and a lot more, use these tools when necessary otherwise answer the question"
    )
    
    # Add system message only if it's not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_message] + messages
    
    # Get response from the model
    response = await model.ainvoke(messages)
    
    return {"messages": [response]}

# Define a Schema where the flow should continue and execute remaining tool calls or end
def should_continue(state: Gideon):
    """Function to control the flow of our Conditional edge"""
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "Continue"
    else:
        return "END"

async def create_graph(model, mcp_tools):
    """Create and compile the graph"""
    # Define the graph flow and conditional edges
    graph = StateGraph(Gideon)
    
    # Create a wrapper function for model_call that includes the model
    async def model_call_wrapper(state: Gideon) -> Gideon:
        return await model_call(state, model)
    
    graph.add_node("Gideon", model_call_wrapper)
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

    return graph.compile()

# Start Main Function code
async def main():
    try:
        # Initialize the agent
        model, mcp_tools = await initialize_agent()
        app = await create_graph(model, mcp_tools)
        
        print("Hello! my name is Gideon! How may I assist you today?")
        
        prompt = ""
        while "exit" not in prompt.lower():
            prompt = input("You: ")
            
            if "exit" in prompt.lower():
                break
            
            # Use the actual user input instead of hardcoded message
            input_state = {"messages": [HumanMessage(content=prompt)]}
            
            try:
                result = await app.ainvoke(input_state)
                
                # Print the final response
                messages = result["messages"]
                for message in messages:
                    if isinstance(message, tuple):
                        print(message)
                    else:
                        message.pretty_print()
                        
            except Exception as e:
                print(f"Error processing request: {e}")
        
        print("Exiting System.... Have a Great day!")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())