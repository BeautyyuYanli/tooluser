import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from openai import AsyncStream
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta, ToolCallChunk, FunctionCallChunk
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.shared_params import FunctionDefinition

from tooluser.tool_user import make_tool_user
from tooluser.hermes_transform import HermesTransformation, tools_list_prompt
from tooluser.transform import Transformation

# --- Fixtures ---

@pytest_asyncio.fixture
async def mock_openai_client():
    client = MagicMock() # Use MagicMock for AsyncOpenAI client itself
    client.chat = MagicMock()
    client.chat.completions = AsyncMock() # Mock the completions attribute with AsyncMock
    # Mock the .create method on the completions mock
    client.chat.completions.create = AsyncMock() 
    return client

@pytest_asyncio.fixture
async def hermes_tool_user_client(mock_openai_client):
    # Ensure the mock_openai_client's methods are awaitable if make_tool_user expects to await them
    # or if the proxy tries to call them with await.
    # make_tool_user itself is synchronous in setup.
    return make_tool_user(client=mock_openai_client, transformation=HermesTransformation())

@pytest.fixture
def sample_messages() -> list[ChatCompletionMessageParam]:
    return [{"role": "user", "content": "Hello"}]

@pytest.fixture
def sample_tools() -> list[FunctionDefinition]:
    return [{
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }]

# --- Helper to create async stream of chunks ---
async def mock_chunk_stream_content_only(content_parts: list[str]) -> AsyncStream[ChatCompletionChunk]:
    for i, part in enumerate(content_parts):
        yield ChatCompletionChunk(
            id=f"chunk_{i}",
            choices=[{
                "delta": ChoiceDelta(content=part, role="assistant", tool_calls=None),
                "index": 0,
                "finish_reason": None,
            }],
            created=12345,
            model="gpt-test",
        )

# --- Tests ---

@pytest.mark.asyncio
async def test_streaming_returns_async_iterable(hermes_tool_user_client, mock_openai_client, sample_messages, sample_tools):
    # Configure the mock's create method to return an empty async stream for this specific test
    mock_openai_client.chat.completions.create.return_value = mock_chunk_stream_content_only([])

    stream_response = await hermes_tool_user_client.chat.completions.create(
        model="gpt-test",
        messages=sample_messages,
        tools=sample_tools,
        stream=True
    )
    
    assert hasattr(stream_response, '__aiter__'), "Response should be an async iterable."
    # To be more specific, check type if AsyncStream is easily comparable or use isinstance
    # For now, checking for __aiter__ is a good start.

@pytest.mark.asyncio
async def test_streaming_hermes_trans_param_messages(hermes_tool_user_client, mock_openai_client, sample_messages, sample_tools):
    # Configure the mock's create method to return an empty async stream
    mock_openai_client.chat.completions.create.return_value = mock_chunk_stream_content_only([])

    await hermes_tool_user_client.chat.completions.create(
        model="gpt-test",
        messages=sample_messages,
        tools=sample_tools,
        stream=True
    )
    
    # Assert that the mock_openai_client.chat.completions.create was called
    mock_openai_client.chat.completions.create.assert_called_once()
    
    # Get the actual arguments passed to the mocked create method
    called_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    
    # Check transformed messages
    expected_system_prompt = tools_list_prompt(sample_tools)
    actual_messages = called_kwargs.get("messages", [])
    
    assert len(actual_messages) == len(sample_messages) + 1, "System prompt should have been added."
    assert actual_messages[0]["role"] == "system"
    assert actual_messages[0]["content"] == expected_system_prompt
    assert actual_messages[1] == sample_messages[0]
    
    # Check that 'tools' param is not passed to underlying call for Hermes
    assert "tools" not in called_kwargs, "'tools' should have been removed from kwargs for Hermes."


# --- Helper to create async stream of chunks with tool calls ---
async def mock_chunk_stream_with_tool_calls() -> AsyncStream[ChatCompletionChunk]:
    # Simulates a sequence: content -> tool call start -> tool call args -> content
    # Tool call: get_weather(location="Boston")
    
    yield ChatCompletionChunk(
        id="chunk_0", choices=[{"delta": ChoiceDelta(content="Thinking... ", role="assistant"), "index": 0}], created=1, model="gpt-test"
    )
    # Start of tool call
    yield ChatCompletionChunk(
        id="chunk_1",
        choices=[{
            "delta": ChoiceDelta(
                role="assistant",
                tool_calls=[ToolCallChunk(index=0, id="call_abc123", function=FunctionCallChunk(name="get_weather", arguments=""), type="function")]
            ), 
            "index": 0
        }],
        created=2, model="gpt-test"
    )
    # Arguments part 1
    yield ChatCompletionChunk(
        id="chunk_2",
        choices=[{
            "delta": ChoiceDelta(
                role="assistant",
                tool_calls=[ToolCallChunk(index=0, id=None, function=FunctionCallChunk(name=None, arguments="{\"location\": "), type="function")]
            ),
            "index": 0
        }],
        created=3, model="gpt-test"
    )
    # Arguments part 2
    yield ChatCompletionChunk(
        id="chunk_3",
        choices=[{
            "delta": ChoiceDelta(
                role="assistant",
                tool_calls=[ToolCallChunk(index=0, id=None, function=FunctionCallChunk(name=None, arguments="\"Boston\"}"), type="function")]
            ),
            "index": 0
        }],
        created=4, model="gpt-test"
    )
    # Mimic end of tool call arguments for index 0, possibly by next chunk having content or different tool index
    # For simplicity, we assume the above chunk completes the arguments for the tool call.
    # A new content chunk implies the previous tool call is done.
    yield ChatCompletionChunk(
        id="chunk_4", choices=[{"delta": ChoiceDelta(content=" Okay.", role="assistant"), "index": 0}], created=5, model="gpt-test"
    )

# --- New Tests ---

@pytest.mark.asyncio
async def test_streaming_hermes_chunk_transformation_content_only(hermes_tool_user_client, mock_openai_client, sample_messages):
    content_parts = ["This is ", "a test."]
    # Configure the mock's create method
    mock_openai_client.chat.completions.create.return_value = mock_chunk_stream_content_only(content_parts)

    stream_response = await hermes_tool_user_client.chat.completions.create(
        model="gpt-test",
        messages=sample_messages,
        # tools not needed if we are only testing content streaming part of completion
        stream=True
    )
    
    collected_chunks = []
    async for chunk in stream_response:
        collected_chunks.append(chunk)
    
    assert len(collected_chunks) == len(content_parts)
    for i, part in enumerate(content_parts):
        assert collected_chunks[i].choices[0].delta.content == part
        assert collected_chunks[i].choices[0].delta.tool_calls is None

@pytest.mark.asyncio
async def test_streaming_hermes_chunk_transformation_with_tool_calls(hermes_tool_user_client, mock_openai_client, sample_messages, sample_tools):
    # This test relies on the specific mock_chunk_stream_with_tool_calls implementation
    # The mock stream has 5 chunks: content, tool_name, tool_arg1, tool_arg2, content
    # Chunks with index 1, 2, 3 (0-indexed) in the mock stream are tool call chunks.
    
    mock_openai_client.chat.completions.create.return_value = mock_chunk_stream_with_tool_calls()

    stream_response = await hermes_tool_user_client.chat.completions.create(
        model="gpt-test",
        messages=sample_messages,
        tools=sample_tools, # tools are provided so trans_param_messages runs
        stream=True
    )
    
    collected_chunks = []
    async for chunk in stream_response:
        collected_chunks.append(chunk)

    assert len(collected_chunks) == 5 # Based on mock_chunk_stream_with_tool_calls

    # Expected states based on HermesTransformation.trans_stream_completion_message
    # Chunk 0 (original: content "Thinking... ")
    assert collected_chunks[0].choices[0].delta.content == "Thinking... "
    assert collected_chunks[0].choices[0].delta.tool_calls is None

    # Chunk 1 (original: tool_calls with name="get_weather")
    assert collected_chunks[1].choices[0].delta.content and "[hermes_tool_calls_conversion_placeholder]" in collected_chunks[1].choices[0].delta.content
    assert collected_chunks[1].choices[0].delta.tool_calls is None
    
    # Chunk 2 (original: tool_calls with arguments="{\"location\": ")
    assert collected_chunks[2].choices[0].delta.content and "[hermes_tool_calls_conversion_placeholder]" in collected_chunks[2].choices[0].delta.content
    assert collected_chunks[2].choices[0].delta.tool_calls is None

    # Chunk 3 (original: tool_calls with arguments="\"Boston\"}")
    assert collected_chunks[3].choices[0].delta.content and "[hermes_tool_calls_conversion_placeholder]" in collected_chunks[3].choices[0].delta.content
    assert collected_chunks[3].choices[0].delta.tool_calls is None

    # Chunk 4 (original: content " Okay.")
    assert collected_chunks[4].choices[0].delta.content == " Okay."
    assert collected_chunks[4].choices[0].delta.tool_calls is None
