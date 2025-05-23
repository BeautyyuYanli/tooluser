from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import print

from tooluser import make_tool_user

load_dotenv()


async def main():
    oai = make_tool_user(AsyncOpenAI())
    res = await oai.chat.completions.create(
        stream=True,
        model="deepseek/deepseek-chat-v3-0324",  # From OpenRouter https://openrouter.ai/deepseek/deepseek-chat-v3-0324
        messages=[
            {
                "role": "user",
                "content": "What's the time in Shanghai? Firstly describe your thinking and then use the tools.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the time in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get the time for",
                            },
                        },
                    },
                },
            }
        ],
    )
    async for chunk in res:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
        if delta.tool_calls:
            print(delta.tool_calls, end="", flush=True)
    print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
