import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv('.env', override=True)


async def mcp_agent():
    client = MultiServerMCPClient(
        {
            "streamable-mcp-server": {
                "url": "http://127.0.0.1:12306/mcp",  # ensure server is running
                "transport": "streamable_http",
            }
        }
    )

    try:
        tools_mcp = await client.get_tools()
        print("Tools:", tools_mcp)

        agent = create_react_agent(
            "groq:llama-3.1-8b-instant",  # or ChatOllama(...) if using Ollama
            tools_mcp
        )

        chrome_response = await agent.ainvoke(
            {"messages": "Play 'Thick of it' the song from KSI on youtube"}
        )

        print("Calls:", chrome_response)

    finally:
        # ✅ this ensures the connection is torn down properly
        await client.close()


if __name__ == "__main__":
    asyncio.run(mcp_agent())
