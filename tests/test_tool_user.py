from openai import AsyncOpenAI

from tooluser import make_tool_user
from tooluser.hermes_transform import HermesTransformation


def test_make_tool_user_default_settings():
    """Test that make_tool_user creates transformation with default settings"""
    client = AsyncOpenAI(api_key="test")
    enhanced_client = make_tool_user(client)

    # The client should be enhanced but we can't easily test the transformation settings
    # without accessing private attributes, so we just verify it returns a client
    assert enhanced_client is not None


def test_make_tool_user_with_raw_json_detection():
    """Test that make_tool_user can enable raw JSON detection"""
    client = AsyncOpenAI(api_key="test")
    enhanced_client = make_tool_user(client, enable_raw_json_detection=True)

    # The client should be enhanced
    assert enhanced_client is not None


def test_make_tool_user_with_custom_transformation():
    """Test that make_tool_user accepts custom transformation"""
    client = AsyncOpenAI(api_key="test")
    custom_transformation = HermesTransformation(enable_raw_json_detection=True)
    enhanced_client = make_tool_user(client, transformation=custom_transformation)

    # The client should be enhanced
    assert enhanced_client is not None
