import json
import re
import uuid
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


class HermesStreamProcessor(StreamProcessor):
    """Processes a stream of text, yielding tool calls and other content."""

    start_tag: str
    end_tag: str
    buffer_size: int
    buffer: str
    in_tool_call: bool

    def __init__(self, start_tag: str, end_tag: str):
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.buffer_size = len(start_tag)
        self.buffer = ""
        self.in_tool_call = False

    def process(self, chunk: str) -> list[StreamOutputType]:
        self.buffer += chunk
        outputs = []
        while True:
            if not self.in_tool_call:
                start_idx = self.buffer.find(self.start_tag)
                if start_idx == -1:
                    # No tool call start found, yield everything up to the last BUFFER_SIZE characters
                    if len(self.buffer) > self.buffer_size:
                        output = self.buffer[: -self.buffer_size]
                        self.buffer = self.buffer[-self.buffer_size :]
                        outputs.append(output)
                        continue
                    else:
                        break
                else:
                    # Found start of tool call
                    output = self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx:]
                    self.in_tool_call = True
                    outputs.append(output)
                    continue
            else:
                # In tool call
                end_idx = self.buffer.find(self.end_tag)
                if end_idx == -1:
                    break
                else:
                    output = self.buffer[: end_idx + len(self.end_tag)]
                    self.buffer = self.buffer[end_idx + len(self.end_tag) :]
                    self.in_tool_call = False
                    outputs.append(tool_call_parse(output))
                    continue
        return outputs

    def finalize(self) -> StreamOutputType:
        if self.in_tool_call:
            try:
                return tool_call_parse(self.buffer)
            except Exception:
                return self.buffer
        else:
            return self.buffer


class HermesTransformation(Transformation):
    """Transform tool_use API call to a user prompt, in Hermes template format.
    ref: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json#L198"""

    @classmethod
    def create_stream_processor(cls) -> StreamProcessor:
        return HermesStreamProcessor(start_tag="<tool_call>", end_tag="</tool_call>")

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
        processor = self.__class__.create_stream_processor()
        if completion.content is not None:
            tool_calls: List[ChatCompletionMessageToolCall] = []
            output_content = ""
            outputs = processor.process(completion.content)
            outputs.append(processor.finalize())
            for output in outputs:
                if isinstance(output, ChatCompletionMessageToolCall):
                    tool_calls.append(output)
                else:
                    output_content += output
            completion.content = output_content
            if tool_calls:
                completion.tool_calls = tool_calls
        return completion

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
