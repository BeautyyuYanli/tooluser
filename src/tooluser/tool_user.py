from functools import wraps

from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion

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
        async def create(self, *args, **kwargs) -> ChatCompletion:  # type: ignore
            messages = kwargs.get("messages", [])
            tools = kwargs.pop("tools", [])
            if kwargs.pop("stream", False):
                raise ValueError("Stream is not supported for tool_user")
            if tools:
                kwargs["messages"] = transformation.trans_param_messages(
                    messages, tools
                )

            response: ChatCompletion = await super().create(*args, **kwargs)
            for choice in response.choices:
                choice.message = transformation.trans_completion_message(choice.message)
            return response

    client.chat.completions = ProxyAsyncCompletions(client=client)  # type: ignore
    return client
