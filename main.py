import openai
from openai import OpenAI, AzureOpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv
import langchain

import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime

  

from clients import chat_client, embedding_client

load_dotenv()

#Loading chroma db from its current location locally - in future this could be an Azure file share or other mounted drive
chroma_client = chromadb.PersistentClient(path = os.getenv('CHROMA_PATH'))
collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')





def main():
    print("Hello from colonoscopy-triage-azure!")


if __name__ == "__main__":
    main()
