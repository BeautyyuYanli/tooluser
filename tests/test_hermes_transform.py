import pytest
from openai.types.chat import ChatCompletionMessageToolCall

from tooluser.hermes_transform import (
    HermesStreamProcessor,
    HermesTransformation,
    tool_call_parse,
)


def test_tool_call_parse_with_tags():
    # Test original behavior with tool_call tags
    text = """<tool_call>
    {
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco",
            "unit": "celsius"
        }
    }
    </tool_call>"""

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


def test_tool_call_parse_with_tags_broken_json():
    # Test original behavior with tool_call tags
    text = """<tool_call>
    {
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco""
            "unit": "celsius"
        }
    }
    </tool_call>"""

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


def test_tool_call_parse_without_tags():
    # Some LLM(Deepseek v3) will occassionally not return the tool_call tags, but return the raw JSON
    text = """{
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco",
            "unit": "celsius"
        }
    }"""

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


def test_tool_call_parse_without_tags_broken_json():
    # Some LLM(Deepseek v3) will occassionally not return the tool_call tags, but return the raw JSON
    text = """{
        "name": "get_weather""
        "arguments": {
            "location": "San Francisco",
            "unit": "celsius"
        }
    }"""

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


def test_tool_call_parse_with_invalid_json():
    # Test with invalid JSON that can be repaired
    text = """{
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco",
            "unit": "celsius"
        }
    """  # Missing closing brace

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


def test_tool_call_parse_invalid_format():
    # Test with invalid format (missing required fields)
    text = """{
        "name": "get_weather"
    }"""  # Missing arguments field

    with pytest.raises(
        ValueError, match="Invalid tool call format - missing required fields"
    ):
        tool_call_parse(text)


def test_tool_call_parse_invalid_json():
    # Test with completely invalid JSON
    text = "this is not json at all"

    with pytest.raises(
        ValueError, match="Invalid tool call format - missing required fields"
    ):
        tool_call_parse(text)


def test_tool_call_parse_with_extra_fields():
    # Test with extra fields in the JSON
    text = """{
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco",
            "unit": "celsius"
        },
        "extra_field": "should be ignored"
    }"""

    result = tool_call_parse(text)
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.type == "function"
    assert result.function.name == "get_weather"
    assert (
        result.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'
    )
    assert result.id.startswith("tool_get_weather_")


# Tests for the enhanced stream processor
def test_stream_processor_raw_json_disabled_by_default():
    # Test that raw JSON detection is disabled by default
    processor = HermesStreamProcessor("<tool_call>", "</tool_call>")

    text = (
        """Some text before {"name": "get_weather", "arguments": {"location": "NYC"}}"""
    )

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    # Should be treated as all text, no tool calls
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    assert len(tool_calls) == 0

    # All content should be text
    text_parts = [o for o in outputs if isinstance(o, str)]
    combined_text = "".join(text_parts)
    assert "get_weather" in combined_text


def test_stream_processor_with_raw_json_enabled():
    # Test that the stream processor can detect raw JSON function calls when enabled
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    text = (
        """Some text before {"name": "get_weather", "arguments": {"location": "NYC"}}"""
    )

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    # Check that we have the expected content
    text_parts = [o for o in outputs if isinstance(o, str)]
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert any("Some text before" in part for part in text_parts)


def test_stream_processor_thinking_aloud():
    # Test LLM thinking aloud before function call
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    text = """I need to check the weather for you. Let me search for that information.
{"name": "get_weather", "arguments": {"location": "San Francisco"}}"""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    text_parts = [o for o in outputs if isinstance(o, str)]
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    combined_text = "".join(text_parts)
    assert "I need to check the weather" in combined_text


def test_stream_processor_avoids_false_positives():
    # Test that legitimate JSON data isn't mistaken for function calls
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    # This should NOT be detected as a function call (data structure, not at end, no action words)
    text = """Here's some user data: {"name": "config_data", "arguments": {"port": 8080, "host": "localhost"}} that we need to process further."""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    # Should be treated as all text, no tool calls
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    assert len(tool_calls) == 0

    # All content should be text
    text_parts = [o for o in outputs if isinstance(o, str)]
    combined_text = "".join(text_parts)
    assert "config_data" in combined_text


def test_stream_processor_function_at_end():
    # Test function call at the end of response (should be detected)
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    text = """The user wants weather information. {"name": "get_weather", "arguments": {"location": "Boston"}}"""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


def test_stream_processor_with_action_function_names():
    # Test function names with action patterns
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    text = """{"name": "search_files", "arguments": {"pattern": "*.py"}}"""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search_files"


def test_stream_processor_unended_tool_call_tag():
    # Test that tool_call tags still take priority over raw JSON
    processor = HermesStreamProcessor("<tool_call>", "</tool_call>")

    text = """I need to call a function. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}"""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    # Should find both calls, but tagged one should come first
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


def test_stream_processor_prioritizes_tool_call_tags():
    # Test that tool_call tags still take priority over raw JSON
    processor = HermesStreamProcessor(
        "<tool_call>", "</tool_call>", enable_raw_json_detection=True
    )

    text = """I need to call a function. <tool_call>{"name": "tagged_call", "arguments": {}}</tool_call> {"name": "get_weather", "arguments": {"location": "NYC"}}"""

    outputs = processor.process(text)
    outputs.append(processor.finalize())

    # Should find both calls, but tagged one should come first
    tool_calls = [o for o in outputs if isinstance(o, ChatCompletionMessageToolCall)]
    expected_call_count = 2  # One tagged call + one raw JSON call
    assert len(tool_calls) == expected_call_count
    assert tool_calls[0].function.name == "tagged_call"  # Tag priority
    assert tool_calls[1].function.name == "get_weather"  # Raw JSON detected after


def test_transformation_class_configuration():
    # Test the transformation class with raw JSON detection enabled
    transformation = HermesTransformation(enable_raw_json_detection=True)

    from openai.types.chat import ChatCompletionMessage

    message = ChatCompletionMessage(
        role="assistant",
        content='I can help with that. {"name": "get_weather", "arguments": {"location": "Boston"}}',
    )

    result = transformation.trans_completion_message(message)

    # Should have extracted the tool call
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.content is not None and "I can help with that." in result.content


def test_transformation_class_default_behavior():
    # Test the transformation class with default settings (raw JSON disabled)
    transformation = HermesTransformation()  # default: enable_raw_json_detection=False

    from openai.types.chat import ChatCompletionMessage

    message = ChatCompletionMessage(
        role="assistant",
        content='I can help with that. {"name": "get_weather", "arguments": {"location": "Boston"}}',
    )

    result = transformation.trans_completion_message(message)

    # Should NOT have extracted the tool call
    assert result.tool_calls is None
    assert result.content is not None and "get_weather" in result.content
