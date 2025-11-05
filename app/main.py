#main file for fast api

import os
from fastapi import FastAPI
import json
from dotenv import load_dotenv
import chromadb
import asyncio


from app.functions import format_query_json, format_query_summary, get_embedding, query_collection, generate_recommendation

chroma_client = chromadb.PersistentClient(path = os.getenv('CHROMA_PATH'))
collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

app = FastAPI(title = "Colonoscopy Triage Recommendation API")

@app.post('/summarize')
async def summarize_query(request: dict):
    user_query = request.get('user_query', '')

    summary = await format_query_summary(user_query)
    json_summary = await format_query_json(user_query)

    return {'summary': summary}, {'json summary': json_summary}

@app.post('/recommend')
async def recommendation(request: dict):
    user_query = request.get('user_query', '')

    summary = await format_query_summary(user_query)
    query_embedding = await get_embedding(summary)
    results = query_collection(query_embedding, collection, n_results = 10)
    rec = await generate_recommendation(results, user_query)

    return {'recommendation': rec}
