from typing import (
    AsyncIterable,
    AsyncIterator,
    Iterable,
    Literal,
    Optional,
    Union,
    overload,
)

from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN, NotGiven
from openai._utils import required_args
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing_extensions import Self

from tooluser.hermes_transform import HermesTransformation
from tooluser.transform import StreamProcessor, Transformation


class _AsyncStreamLike(AsyncStream[ChatCompletionChunk]):
    """Wrapper that provides the same interface as OpenAI's AsyncStream"""

    def __init__(self, stream: AsyncIterable[ChatCompletionChunk]):
        # Store the wrapped stream
        self._iterator = aiter(stream)

    async def __anext__(self) -> ChatCompletionChunk:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        async for item in self._iterator:
            yield item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


def make_tool_user(
    client: AsyncOpenAI,
    transformation: Transformation | None = None,
    enable_raw_json_detection: bool = False,
):
    if transformation is None:
        transformation = HermesTransformation(
            enable_raw_json_detection=enable_raw_json_detection
        )

    class ProxyAsyncCompletions(AsyncCompletions):
        def __init__(self, client):
            # Copy all attributes from the parent AsyncCompletions instance
            self._client = client
            super().__init__(client)

        @overload
        async def create(
            self,
            *,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, object],
            stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
            **kwargs,
        ) -> ChatCompletion: ...

        @overload
        async def create(
            self,
            *,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, object],
            stream: Literal[True],
            **kwargs,
        ) -> _AsyncStreamLike: ...

        @overload
        async def create(
            self,
            *,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, object],
            stream: bool,
            **kwargs,
        ) -> ChatCompletion | _AsyncStreamLike: ...

        @required_args(["messages", "model"], ["messages", "model", "stream"])
        async def create(self, *args, **kwargs):
            messages = kwargs.get("messages", [])
            tools = kwargs.pop("tools", [])
            stream = kwargs.get("stream", False)
            if tools:
                kwargs["messages"] = transformation.trans_param_messages(
                    messages, tools
                )
            if not stream:
                response: ChatCompletion = await super().create(*args, **kwargs)
                for choice in response.choices:
                    choice.message = transformation.trans_completion_message(
                        choice.message
                    )
                return response
            else:
                response_stream: AsyncIterable[
                    ChatCompletionChunk
                ] = await super().create(*args, **kwargs)  # type: ignore

                async def _wrapped():
                    processors: dict[int, StreamProcessor] = {}
                    async for chunk in response_stream:
                        for idx, choice in enumerate(chunk.choices):
                            if idx not in processors:
                                processors[idx] = (
                                    transformation.create_stream_processor_instance()
                                )
                            if choice.delta.content is not None:
                                if choice.finish_reason is None:
                                    choice.delta = (
                                        transformation.trans_completion_message_stream(
                                            processors[idx], delta=choice.delta
                                        )
                                    )
                                else:
                                    choice.delta = (
                                        transformation.trans_completion_message_stream(
                                            processors[idx],
                                            delta=choice.delta,
                                            finalize=True,
                                        )
                                    )
                        for choice in chunk.choices:
                            # Omit empty chunk
                            if (
                                (choice.finish_reason is None)
                                and (not choice.delta.content)
                                and (not choice.delta.tool_calls)
                            ):
                                pass
                            else:
                                yield chunk
                                break

                return _AsyncStreamLike(_wrapped())

    client.chat.completions = ProxyAsyncCompletions(client=client)  # type: ignore
    return client
