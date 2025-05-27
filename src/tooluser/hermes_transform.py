import json
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List

from jinja2 import Template
from json_repair import repair_json
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from tooluser.transform import StreamOutputType, StreamProcessor, Transformation


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

    # Check if the parsed data has the required structure for a function call
    if not isinstance(tool_call_data, dict) or not all(
        key in tool_call_data for key in ["name", "arguments"]
    ):
        raise ValueError("Invalid tool call format - missing required fields")

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


# Helper functions for the processing logic


def _is_potential_function_call_start(text: str, start_pos: int) -> bool:
    """Check if the JSON starting at start_pos looks like a function call."""
    # More restrictive pattern: double quotes only, arguments must be object/array
    remaining = text[start_pos:]
    pattern = (
        r'^\s*\{\s*"name"\s*:\s*"[a-zA-Z_][a-zA-Z0-9_]*"\s*,\s*"arguments"\s*:\s*[\{\[]'
    )

    if not re.match(pattern, remaining, re.DOTALL):
        return False

    # Additional heuristics to reduce false positives
    return _passes_function_call_heuristics(text, start_pos)


def _passes_function_call_heuristics(text: str, start_pos: int) -> bool:
    """Apply additional heuristics to determine if this is likely a function call."""

    # Only detect JSON that appears at the very end of the text
    # This catches the common case where LLMs output: "I'll help with that. {JSON}"
    json_end = _find_json_end(text, start_pos)
    if json_end != -1:
        after_json = text[json_end:].strip()
        # Must be at the end with only whitespace after
        return len(after_json) == 0

    return False


def _find_json_end(text: str, start_pos: int) -> int:
    """Find the end of a JSON object starting at start_pos, returning the position after the closing brace."""
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return i + 1

    return -1  # No matching closing brace found


class HermesStreamProcessor(StreamProcessor):
    """Processes a stream of text, yielding tool calls and other content."""

    start_tag: str
    end_tag: str
    buffer_size: int
    buffer: str
    in_tool_call: bool
    in_raw_json: bool
    enable_raw_json_detection: bool

    def __init__(
        self, start_tag: str, end_tag: str, enable_raw_json_detection: bool = False
    ):
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.buffer_size = len(start_tag)
        self.buffer = ""
        self.in_tool_call = False
        self.in_raw_json = False
        self.enable_raw_json_detection = enable_raw_json_detection

    def process(self, chunk: str) -> list[StreamOutputType]:
        self.buffer += chunk
        outputs = []

        while True:
            if not self.in_tool_call and not self.in_raw_json:
                # Look for tool_call tags first (higher priority)
                start_idx = self.buffer.find(self.start_tag)

                # Look for potential raw JSON function calls (only if enabled)
                json_start_idx = -1
                if self.enable_raw_json_detection:
                    for i in range(len(self.buffer)):
                        if self.buffer[i] == "{" and _is_potential_function_call_start(
                            self.buffer, i
                        ):
                            json_start_idx = i
                            break

                # Decide which pattern to follow
                if start_idx != -1 and (
                    json_start_idx == -1 or start_idx < json_start_idx
                ):
                    # Found tool_call tag first or only tool_call tag
                    output = self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx:]
                    self.in_tool_call = True
                    if output:
                        outputs.append(output)
                    continue

                elif json_start_idx != -1:
                    # Found potential raw JSON function call
                    output = self.buffer[:json_start_idx]
                    self.buffer = self.buffer[json_start_idx:]
                    self.in_raw_json = True
                    if output:
                        outputs.append(output)
                    continue

                # No patterns found, yield everything up to the last BUFFER_SIZE characters
                elif len(self.buffer) > self.buffer_size:
                    output = self.buffer[: -self.buffer_size]
                    self.buffer = self.buffer[-self.buffer_size :]
                    outputs.append(output)
                    continue
                else:
                    break

            elif self.in_tool_call:
                # In tool call, look for end tag
                end_idx = self.buffer.find(self.end_tag)
                if end_idx == -1:
                    break
                else:
                    output = self.buffer[: end_idx + len(self.end_tag)]
                    self.buffer = self.buffer[end_idx + len(self.end_tag) :]
                    self.in_tool_call = False
                    try:
                        outputs.append(tool_call_parse(output))
                    except Exception:
                        # If parsing fails, treat as regular text
                        outputs.append(output)
                    continue

            elif self.in_raw_json:
                # In raw JSON, look for the end of the JSON object
                json_end = _find_json_end(self.buffer, 0)
                if json_end == -1:
                    # Haven't found the end yet, need more data
                    break
                else:
                    output = self.buffer[:json_end]
                    self.buffer = self.buffer[json_end:]
                    self.in_raw_json = False
                    try:
                        # Try to parse as function call
                        parsed_call = tool_call_parse(output)
                        outputs.append(parsed_call)
                    except Exception:
                        # If parsing fails, treat as regular text
                        outputs.append(output)
                    continue

        return outputs

    def finalize(self) -> StreamOutputType:
        if self.in_tool_call or self.in_raw_json:
            try:
                return tool_call_parse(self.buffer)
            except Exception:
                return self.buffer
        else:
            return self.buffer


@dataclass
class HermesTransformation(Transformation):
    """Transform tool_use API call to a user prompt, in Hermes template format.
    ref: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json#L198"""

    enable_raw_json_detection: bool = False

    def create_stream_processor(self) -> StreamProcessor:
        return HermesStreamProcessor(
            start_tag="<tool_call>",
            end_tag="</tool_call>",
            enable_raw_json_detection=self.enable_raw_json_detection,
        )

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
        message: ChatCompletionMessage,
    ) -> ChatCompletionMessage:
        processor = self.create_stream_processor()
        if message.content is not None:
            tool_calls: List[ChatCompletionMessageToolCall] = []
            output_content = ""
            outputs = processor.process(message.content)
            outputs.append(processor.finalize())
            for output in outputs:
                if isinstance(output, ChatCompletionMessageToolCall):
                    tool_calls.append(output)
                else:
                    output_content += output
            message.content = output_content
            if tool_calls:
                message.tool_calls = tool_calls
        return message

    def trans_completion_message_stream(
        self,
        processor: StreamProcessor,
        delta: ChoiceDelta,
        finalize: bool = False,
    ) -> ChoiceDelta:
        if not finalize:
            if delta.content is None:
                raise ValueError("Delta content is None but finalize is False")
            outputs = processor.process(delta.content)
        else:
            outputs = [processor.finalize()]
        tool_calls: list[ChatCompletionMessageToolCall] = []
        content = ""
        for output in outputs:
            if isinstance(output, ChatCompletionMessageToolCall):
                tool_calls.append(output)
            else:
                content += output
        delta.content = content or None
        delta.tool_calls = [
            ChoiceDeltaToolCall(
                index=0,
                id=x.id,
                function=ChoiceDeltaToolCallFunction(
                    name=x.function.name,
                    arguments=x.function.arguments,
                ),
                type=x.type,
            )
            for x in tool_calls
        ] or None
        return delta
