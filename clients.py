from openai import OpenAI, AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chat_client = AzureOpenAI(
    api_version = os.getenv('AZURE_GPT_API_VERSION'),
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('AZURE_ENDPOINT'),
)

embedding_client = AzureOpenAI(
    api_version = os.getenv('AZURE_EMBEDDING_API_VERSION'),
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('AZURE_ENDPOINT'),
)

