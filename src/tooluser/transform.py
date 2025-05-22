from typing import Iterable, Protocol, Union

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.shared_params.function_definition import FunctionDefinition

StreamOutputType = Union[str, ChatCompletionMessageToolCall]


class StreamProcessor(Protocol):
    def process(self, chunk: str) -> list[StreamOutputType]: ...
    def finalize(self) -> StreamOutputType: ...


class Transformation(Protocol):
    @classmethod
    def create_stream_processor(cls) -> StreamProcessor: ...

    def trans_param_messages(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Iterable[FunctionDefinition],
    ) -> Iterable[ChatCompletionMessageParam]: ...

    def trans_completion_message(
        self,
        message: ChatCompletionMessage,
    ) -> ChatCompletionMessage: ...

    def trans_completion_message_stream(
        self,
        processor: StreamProcessor,
        delta: ChoiceDelta,
        finalize: bool = False,
    ) -> ChoiceDelta: ...
