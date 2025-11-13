import openai
from openai import OpenAI, AzureOpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv
import langchain
import yaml
from pathlib import Path

import asyncio
import requests

import random

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime

  

from app.clients import chat_client, embedding_client

load_dotenv()

BASE_PATH = Path(__file__).parent
PROMPT_PATH = BASE_PATH/'prompts'


async def format_query_json(user_query: str) -> dict: 
    PROMPT_FILE = PROMPT_PATH/'json_summary_prompt.yaml'
    with open(PROMPT_FILE, 'r', encoding = 'utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise Exception(f'Error loading YAML prompt file: {e}')
        system_prompt = f'{config['prompt']['content']}'
        if 'rules' in config:
            rules_text = '\nRules:\n' + '\n'.join(f'- {rule}' for rule in config['rules'])
            system_prompt = f'{system_prompt}\n{rules_text}'
        
        

    user_prompt = f'Please format this medical text into structured JSON output - {user_query}'

    response1 = await chat_client.responses.create(
        model = 'gpt-5-mini',
        text = {'format': {'type': 'json_object'}},
        input = [
            {
                'role':'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            }
        ],
        

    )

    try:
        raw_output = response1.output_text
        result_json = json.loads(raw_output)
        return result_json
    except json.JSONDecodeError:
        return {'error': 'Failed to parse JSON', 'raw_output': response1.output_text}

    


async def format_query_summary(user_query: str) -> str:
    PROMPT_FILE = PROMPT_PATH/'summary_prompt.yaml'
    with open(PROMPT_FILE, 'r', encoding = 'utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise Exception(f'Error loading YAML prompt file: {e}')
        system_prompt = f'{config['prompt']['content']}'
        if 'rules' in config:
            rules_text = '\nRules:\n' + '\n'.join(f'- {rule}' for rule in config['rules'])
            system_prompt = f'{system_prompt}\n{rules_text}'
    

    response = await chat_client.responses.create(
        model = 'gpt-5-mini',
        input = [
            {'role': 'system',
             'content': system_prompt,
             },
             {'role': 'user',
              'content': user_query}
              
        ],
        
    )

    return response.output_text
    

async def get_embedding(text: str) -> List[float]:
    response = await embedding_client.embeddings.create(
        input = text,
        model = 'text-embedding-3-small',
    )
    return response.data[0].embedding


def query_collection(embedding: List[float], collection, n_results: int = 10) -> List[dict]:
    results = collection.query(
        query_embeddings = [embedding],
        n_results = n_results,
    )

    if results['documents']:
        db_output = [
            {'document': doc,
             'metadata': meta,}
             for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
        return db_output
    else:
        return []

async def generate_recommendation(db_result: List[dict], user_query: str) -> str:
    PROMPT_FILE = PROMPT_PATH/'recommendation_prompt.yaml'
    with open (PROMPT_FILE, 'r', encoding = 'utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise Exception(f'Error loading YAML prompt file: {e}')
        
        system_prompt = f'{config['prompt']['content']}'
        if 'rules' in config:
            rules_text = '\nRules:\n' + '\n'.join(f'- {rule}' for rule in config['rules'])

            system_prompt = f'{system_prompt}\n{rules_text}'
        
    db_docs = "".join(db_result[i]['document'] for i in range(len(db_result)))
    response = await chat_client.responses.create(
        model = 'gpt-5-mini',
        input = [
            {'role': 'system',
             'content': system_prompt
             },
            {'role': 'user',
             'content': user_query
             },
            {'role': 'user',
             'content': db_docs}
        ]
    )

    return response.output_text

async def send_request(report_text: str, api_url: str):
    '''
    Sends a free text reort to the API endpoint and returns a recommendation based on the report and database contents
    '''

    data = {'user_query': report_text}

    response = requests.post(api_url, json = data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage()
        }
        if hasattr(record, 'extra_data'):
            log_record['extra'] = record.extra_data
        
        return json.dumps(log_record)

logger = logging.getLogger('colonoscopy_triage_api_logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.propagate = False