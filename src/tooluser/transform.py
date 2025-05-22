from typing import AsyncIterable, Iterable, Protocol

from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition


class Transformation(Protocol):
    def trans_param_messages(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Iterable[FunctionDefinition],
    ) -> Iterable[ChatCompletionMessageParam]: ...

    def trans_completion_message(
        self,
        completion: ChatCompletionMessage,
    ) -> ChatCompletionMessage: ...

    def trans_stream_param_messages(
        self,
        messages: AsyncIterable[ChatCompletionMessageParam],
        tools: Iterable[FunctionDefinition],
    ) -> AsyncIterable[ChatCompletionMessageParam]: ...

    def trans_stream_completion_message(
        self,
        completion_chunk: ChatCompletionChunk,
    ) -> ChatCompletionChunk: ...
