import pytest 
import asyncio

from openai import AsyncAzureOpenAI 
from unittest.mock import AsyncMock, patch

from app.functions import format_query_summary, format_query_json, get_embedding, query_collection, generate_recommendation


@pytest.mark.asyncio
async def test_format_query_summary():
    fake_response = AsyncMock()

    fake_response.output_text = 'Test summary'

    with patch('app.functions.chat_client.responses.create', return_value = fake_response):
        result = await format_query_summary('Test user query')
        assert result == 'Test summary'


