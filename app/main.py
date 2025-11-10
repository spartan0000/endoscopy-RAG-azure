#main file for fast api

import os
from fastapi import FastAPI, Depends, HTTPException, status, Header
import json
from dotenv import load_dotenv
import chromadb
import asyncio
from pydantic import BaseModel
import logging
import sys

load_dotenv()



from app.functions import format_query_json, format_query_summary, get_embedding, query_collection, generate_recommendation, logger

chroma_client = chromadb.PersistentClient(path = './data/chroma_db')
collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')



app = FastAPI(title = "Colonoscopy Triage Recommendation API")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv('MY_API_KEY'):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = 'Invalid or missing API Key'
        )

class SummarizeRequest(BaseModel):
    user_query: str

@app.post('/summarize', dependencies = [Depends(verify_api_key)])
async def summarize_query(request: SummarizeRequest):
    user_query = request.user_query

    summary = await format_query_summary(user_query)
    

    return {'summary': summary}

@app.post('/json_summary', dependencies = [Depends(verify_api_key)])
async def json_summary(request: SummarizeRequest):
    user_query = request.user_query
    json_summary = await format_query_json(user_query)
    return {'json_summary': json_summary}


@app.post('/recommend', dependencies = [Depends(verify_api_key)])
async def recommendation(request: SummarizeRequest):
    user_query = request.user_query

    summary = await format_query_summary(user_query)
    

    query_embedding = await get_embedding(summary)
    results = query_collection(query_embedding, collection, n_results = 10)
    document_contents = [r['document'][:200] for r in results]
    rec = await generate_recommendation(results, user_query)
    #the results variable contains the documents and metadatas retrieved from the database
    #
    logger.info("User input received and recommendation generated",
                 extra = {
                     "extra_data": {
                         'user_query': user_query, 
                         'documents': results, 
                         'document_contents': document_contents,
                         'recommendation': rec,
                         }
                 }
        )
    

    return {'recommendation': rec}
 