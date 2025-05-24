#!/usr/bin/env python3

import asyncio

from openai import AsyncOpenAI

from tooluser import make_tool_user


async def main():
    # Enable raw JSON detection for LLMs that sometimes forget <tool_call> tags
    client = make_tool_user(
        AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="your-api-key",  # Replace with your actual API key
        ),
        enable_raw_json_detection=True,  # Enable the new feature
    )

    # This would work with LLMs that output raw JSON without <tool_call> tags
    response = await client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get the weather for",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )

    message = response.choices[0].message
    print(f"Content: {message.content}")

    if message.tool_calls:
        print("Tool calls detected:")
        for tool_call in message.tool_calls:
            print(f"  - {tool_call.function.name}({tool_call.function.arguments})")
    else:
        print("No tool calls detected")


if __name__ == "__main__":
    asyncio.run(main())
