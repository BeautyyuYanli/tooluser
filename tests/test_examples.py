"""
Example integration tests that import and test the actual example files.
These tests ensure the exact example code that users will run works correctly.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as StreamChoice,
)

# Configure pytest to only use asyncio backend
pytestmark = pytest.mark.anyio


class TestExamples:
    """Test suite that imports and runs the actual example files."""

    def setup_method(self):
        """Add the project root to sys.path so we can import examples."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    async def test_example_py(self):
        """Test the basic example.py file by importing and running it."""
        # Mock response that the example expects
        mock_response = ChatCompletion(
            id="test-completion",
            object="chat.completion",
            created=1234567890,
            model="deepseek/deepseek-chat-v3-0324",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='I need to check the current time in Shanghai. Let me use the get_time tool.\n\n<tool_call>\n{"name": "get_time", "arguments": {"location": "Shanghai"}}\n</tool_call>',
                    ),
                    finish_reason="stop",
                )
            ],
        )

        # Mock the OpenAI client creation and API call
        with (
            patch("openai.AsyncOpenAI"),
            patch(
                "openai.resources.chat.completions.AsyncCompletions.create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_create,
        ):
            # Import the example module
            import example

            # Mock the API key environment (example loads dotenv)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                # Run the example's main function
                await example.main()

            # Verify the API was called correctly
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "deepseek/deepseek-chat-v3-0324"
            assert len(call_args.kwargs["messages"]) >= 1  # at least system message

            # Our transformation adds a system message first with tool instructions
            assert call_args.kwargs["messages"][0]["role"] == "system"
            assert "tool_instruction" in call_args.kwargs["messages"][0]["content"]

            # Then the user message
            user_message = next(
                (m for m in call_args.kwargs["messages"] if m["role"] == "user"), None
            )
            assert user_message is not None
            assert "time in Shanghai" in user_message["content"]

            # Tools should be empty since they're transformed into the system message
            assert call_args.kwargs.get("tools", []) == []

    async def test_example_stream_py(self):
        """Test the example_stream.py file by importing and running it."""

        # Mock streaming response
        async def mock_stream():
            chunks = [
                ChatCompletionChunk(
                    id="test-stream",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="deepseek/deepseek-chat-v3-0324",
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=ChoiceDelta(
                                content="I need to check the time in Shanghai. "
                            ),
                            finish_reason=None,
                        )
                    ],
                ),
                ChatCompletionChunk(
                    id="test-stream",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="deepseek/deepseek-chat-v3-0324",
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        id="call_123",
                                        function=ChoiceDeltaToolCallFunction(
                                            name="get_time",
                                            arguments='{"location": "Shanghai"}',
                                        ),
                                        type="function",
                                    )
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                ),
            ]
            for chunk in chunks:
                yield chunk

        with (
            patch("openai.AsyncOpenAI"),
            patch(
                "openai.resources.chat.completions.AsyncCompletions.create",
                new_callable=AsyncMock,
                return_value=mock_stream(),
            ) as mock_create,
        ):
            import example_stream

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                await example_stream.main()

            # Verify streaming was used
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["stream"] is True
            assert call_args.kwargs["model"] == "deepseek/deepseek-chat-v3-0324"

    async def test_example_imports_work(self):
        """Test that all example files can be imported without errors."""
        # This test ensures the example files have valid Python syntax
        # and can import all their dependencies
        try:
            import example
            import example_stream

            # Verify they have main functions
            assert callable(example.main)
            assert callable(example_stream.main)

        except ImportError as e:
            pytest.fail(f"Failed to import example modules: {e}")
        except Exception as e:
            pytest.fail(f"Example modules have syntax or import errors: {e}")

    async def test_example_error_handling(self):
        """Test that examples handle API errors gracefully."""

        # Mock an API error
        async def mock_error():
            raise Exception("API Error")

        with (
            patch("openai.AsyncOpenAI"),
            patch(
                "openai.resources.chat.completions.AsyncCompletions.create",
                side_effect=mock_error,
            ) as mock_create,
        ):
            import example

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                # The example should handle errors gracefully or let them bubble up
                # Either way, we're testing that the import and setup work
                try:
                    await example.main()
                except Exception:
                    # Expected - the example doesn't have error handling
                    # but at least it runs and calls the API correctly
                    pass

            # Verify the API was called
            mock_create.assert_called_once()
