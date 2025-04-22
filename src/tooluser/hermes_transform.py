import json
import re
import uuid
from typing import Callable, Iterable, List

from jinja2 import Template
from json_repair import repair_json
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from tooluser.transform import Transformation


def tools_list_prompt(tools: Iterable[FunctionDefinition]):
    tools_template = """
<tool_instruction>
You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
<tools>
{{tools}}
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
"""
    return Template(tools_template).render(
        tools=[
            json.dumps(
                tool,
                ensure_ascii=False,
            )
            for tool in tools
        ]
    )


def tool_call_parse(text: str):
    # First check if the text is wrapped in tool_call tags
    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if tool_call_match:
        text = tool_call_match.group(1).strip()

    # Parse the JSON-formatted tool call
    try:
        tool_call_data: dict = repair_json(text, return_objects=True)  # type: ignore
    except Exception as e:
        raise ValueError("Invalid tool call format - must be valid JSON") from e

    try:
        # Create a Function object
        function = Function(
            name=tool_call_data["name"],
            arguments=json.dumps(tool_call_data["arguments"], ensure_ascii=False),
        )

        # Create and return a ChatCompletionMessageToolCall
        return ChatCompletionMessageToolCall(
            id="tool_" + function.name + "_" + uuid.uuid4().hex[:8],
            function=function,
            type="function",
        )
    except KeyError as e:
        raise ValueError("Invalid tool call format - missing required fields") from e


def tool_call_parse_parama(text: str) -> ChatCompletionMessageToolCallParam:
    tool_call = tool_call_parse(text)
    return tool_call.model_dump()  # type: ignore


def tool_call_serialize(tool_call: ChatCompletionMessageToolCallParam):
    # Parse the arguments string back into a dictionary
    try:
        arguments: dict | str = repair_json(
            tool_call["function"]["arguments"], return_objects=True
        )  # type: ignore
    except Exception as e:
        arguments = tool_call["function"]["arguments"]
        raise ValueError("Invalid tool call format - must be valid JSON") from e

    # Create the JSON structure as specified in tools_list_prompt
    tool_call_data = {
        "name": tool_call["function"]["name"],
        "id": tool_call["id"],
        "arguments": arguments,
    }

    return f"""<tool_call>
{json.dumps(tool_call_data, ensure_ascii=False)}
</tool_call>"""


def tool_result_serialize(tool_result: ChatCompletionToolMessageParam):
    res = tool_result["content"]
    if not isinstance(res, str):
        res = "".join([part["text"] for part in res])
    return f"""<tool_result>
<id>{tool_result["tool_call_id"]}</id>
<result>
{res}
</result>
</tool_result>"""


def tool_result_parse(text: str) -> ChatCompletionToolMessageParam:
    id_match = re.search(r"<id>(.*?)</id>", text, re.DOTALL)
    result_match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if not id_match or not result_match:
        raise ValueError("Invalid tool result format")
    return {
        "role": "tool",
        "tool_call_id": id_match.group(1).strip(),
        "content": result_match.group(1).strip(),
    }


def stream_process(
    text_stream: Iterable[str],
    start_tag: str,
    end_tag: str,
    callback: Callable[[str], None],
):
    BUFFER_SIZE = len(start_tag)

    buffer = ""
    in_tool_call = False

    for chunk in text_stream:
        buffer += chunk

        # Keep processing while we can find complete tool calls
        while True:
            if not in_tool_call:
                # Look for the start of a tool call
                start_idx = buffer.find(start_tag)
                if start_idx == -1:
                    # No tool call start found, yield everything up to the last BUFFER_SIZE characters
                    if len(buffer) > BUFFER_SIZE:
                        yield buffer[:-BUFFER_SIZE]
                        buffer = buffer[-BUFFER_SIZE:]
                    break

                # Found start of tool call
                if start_idx > 0:
                    yield buffer[:start_idx]
                buffer = buffer[start_idx:]
                in_tool_call = True

            else:
                # Look for the end of the tool call
                end_idx = buffer.find(end_tag)
                if end_idx == -1:
                    break

                # Found complete tool call
                end_idx += len(end_tag)
                callback(buffer[:end_idx])
                buffer = buffer[end_idx:]
                in_tool_call = False

    # Yield any remaining content
    if buffer:
        yield buffer


def stream_process_tool_call(text_stream: Iterable[str]):
    tool_calls: List[ChatCompletionMessageToolCall] = []

    def callback(text: str):
        tool_calls.append(tool_call_parse(text))

    return (
        stream_process(text_stream, "<tool_call>", "</tool_call>", callback),
        tool_calls,
    )


class HermesTransformation(Transformation):
    """Transform tool_use API call to a user prompt, in Hermes template format.
    ref: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json#L198"""

    def trans_param_messages(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Iterable[FunctionDefinition],
    ) -> Iterable[ChatCompletionMessageParam]:
        new_messages = []
        new_messages.append(
            {
                "role": "system",
                "content": tools_list_prompt(tools),
            }
        )
        for message in messages:
            if "tool_calls" in message:
                new_message = message.copy()
                new_message.pop("tool_calls")
                tools_prompt = [
                    tool_call_serialize(tool_call)
                    for tool_call in message["tool_calls"]
                ]
                content = message.get("content", "")
                if isinstance(content, str) or (content is None):
                    content = content or ""
                    new_message["content"] = content + "\n" + "\n".join(tools_prompt)
                else:
                    new_message["content"] = [
                        *content,
                        *[{"text": t, "type": "text"} for t in tools_prompt],
                    ]
                new_messages.append(new_message)
            elif message["role"] == "tool":
                tool_results = tool_result_serialize(message)
                new_messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )
            else:
                new_messages.append(message)

        return new_messages

    def trans_completion_message(
        self,
        completion: ChatCompletionMessage,
    ) -> ChatCompletionMessage:
        if completion.content is not None:
            streaming, tool_calls = stream_process_tool_call([completion.content])
            completion.content = "".join(streaming)
            if tool_calls:
                completion.tool_calls = tool_calls
        return completion
