# tooluser - Enable tool-use ability for any LLM model (DeepSeek V3/R1, etc.)

For some models/providers that doesn't natively support function calling (e.g. DeepSeek V3/R1), you can use this library to transform the tool calls to a user prompt, in Hermes template format by default.

```
pip install tooluser
```
```python
from openai import AsyncOpenAI
from tooluser import make_tool_user

oai = make_tool_user(AsyncOpenAI())

res = await oai.chat.completions.create(
    model="deepseek/deepseek-chat-v3-0324", # From OpenRouter https://openrouter.ai/deepseek/deepseek-chat-v3-0324
    messages=[{"role": "user", "content": "What's the time in Shanghai?"}],
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
```

Check out the [example.py](example.py) for a runnable example.

## Streaming Support

Yes, this library also supports streaming.

Check out the [example_stream.py](example_stream.py) for a runnable example.

(LLM output for tool using is not streamed, because we use json-repair for it.)

## What's Hermes template?

Function calling is implicitly a prompt template, to make the model understand how to output the structured response as we want. Hermes template is a widely adopted prompt template for function calling.

- [Qwen2.5 - Function Calling Templates](https://qwen.readthedocs.io/en/latest/framework/function_call.html#qwen2-5-function-calling-templates)
- [Qwen2.5-0.5B-Instruct - Tokenizer Config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json#L198)
- [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling#prompt-format-for-function-calling)

## What happens under the hood?

As we want to make use of the OpenAI chat completion API, we do not directly use Hermes template to generate the LLM instruction, but we generate the Hermes style system prompt and user prompt.

The actually API call is:

System:
```
<tool_instruction>
You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
<tools>
['{"type": "function", "function": {"name": "get_time", "description": "Get the time in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the time for"}}}}}']
</tools>

For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call>

Here is an example of a tool call:
<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}
</tool_call>

</tool_instruction>
```

User:
```
What's the time in Shanghai?
```

Assistant:
```
<tool_call>
{"name": "get_time", "arguments": {"location": "Shanghai"}}
</tool_call>
```