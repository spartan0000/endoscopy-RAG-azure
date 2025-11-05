import openai
from openai import OpenAI, AzureOpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv
import langchain

from fastapi import FastAPI


import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime

  

from app.clients import chat_client, embedding_client
from app.functions import format_query_json, format_query_summary, get_embedding, query_collection, generate_recommendation


load_dotenv()

#Loading chroma db from its current location locally - in future this could be an Azure file share or other mounted drive
chroma_client = chromadb.PersistentClient(path = os.getenv('CHROMA_PATH'))
collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

logging.basicConfig(
    filename = 'logs/audit_logs.jsonl',
    level = logging.INFO,
    format = '%(message)s'
)

def log_entry(entry: dict):
    logging.info(json.dumps(entry, ensure_ascii=False))


### sample patient data
with open('data/sample_patient_report_1.txt', 'r', encoding = 'utf-8') as f:
    user_query = f.read()
###

async def main():
    output = await format_query_json(user_query)
    summary = await format_query_summary(user_query)
    query_embedding = await get_embedding(summary)
    results = query_collection(query_embedding, collection, 10)
    recommendation = await generate_recommendation(results, user_query)

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'user_query': user_query,
        'formatted_summary': output,
        'database_results': results,
        'protocol_docs': [r['document'][:200] for r in results],
        'recommendation': recommendation,

    }

    log_entry(log_data)
    print(recommendation)


if __name__ == "__main__":
    asyncio.run(main())
