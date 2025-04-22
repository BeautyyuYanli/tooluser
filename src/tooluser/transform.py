from typing import Iterable, Protocol

from openai.types.chat import (
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
