from tooluser.hermes_transform import HermesTransformation
from tooluser.tool_user import make_tool_user
from tooluser.transform import Transformation

__all__ = ["HermesTransformation", "Transformation", "make_tool_user"]


async def main():
    from openai import AsyncOpenAI

    oai = make_tool_user(AsyncOpenAI())
    res = await oai.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Shanghai?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to get the weather for",
                            },
                        },
                    },
                },
            }
        ],
    )

    print(res.choices[0].message.tool_calls)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
