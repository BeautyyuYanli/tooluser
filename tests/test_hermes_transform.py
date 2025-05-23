import pytest
from openai.types.chat import ChatCompletionMessageToolCall

from tooluser.hermes_transform import tool_call_parse


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
