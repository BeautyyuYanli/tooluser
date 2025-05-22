from functools import wraps
from typing import AsyncGenerator, Union

from openai import AsyncOpenAI, AsyncStream
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from tooluser.hermes_transform import HermesTransformation
from tooluser.transform import Transformation


def make_tool_user(client: AsyncOpenAI, transformation: Transformation | None = None):
    if transformation is None:
        transformation = HermesTransformation()

    class ProxyAsyncCompletions(AsyncCompletions):
        def __init__(self, client):
            # Copy all attributes from the parent AsyncCompletions instance
            self._client = client
            super().__init__(client)

        @wraps(AsyncCompletions.create)
        async def create(
            self, *args, **kwargs
        ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:  # type: ignore
            messages = kwargs.get("messages", [])
            tools_param = kwargs.pop("tools", [])
            is_streaming = kwargs.get("stream", False)

            if tools_param:
                kwargs["messages"] = transformation.trans_param_messages(
                    messages, tools_param
                )
            
            if is_streaming:
                async def stream_transformer() -> AsyncStream[ChatCompletionChunk]:
                    raw_stream = await super().create(*args, **kwargs)
                    async for chunk in raw_stream:
                        transformed_chunk = transformation.trans_stream_completion_message(
                            chunk
                        )
                        yield transformed_chunk
                return stream_transformer()
            else:
                response: ChatCompletion = await super().create(*args, **kwargs)
                for choice in response.choices:
                    choice.message = transformation.trans_completion_message(
                        choice.message
                    )
                return response

    client.chat.completions = ProxyAsyncCompletions(client=client)  # type: ignore
    return client
