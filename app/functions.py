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
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime

  

from app.clients import chat_client, embedding_client

load_dotenv()

async def format_query_json(user_query: str) -> dict: 
    system_prompt = """
    summarize the user input that includes medical data on a person's history of colonoscopy procedures and the pathology reports from the polyps that were removed.
    key information includes the following:
    - dates of the procedure including month, day, and year.
    - the number of polyps
    - the size of the polyps which is generally reported in millimeters - less than 10mm or greater than or equal to 10mm is a useful cutoff
    - the histology of the polyps.  
    - for adenomas - the number of polyps and the size of the largest adenoma, whether there is high grade dysplasia (yes/no), whether the adenoma is tubulovillous or villous (yes/no)
    - for sessile serrated polyps - the number of sessile serrated polyps, the size of the largest sessile serrated polyp, whether there is dysplasia (yes/no)
    - for hyperplastic polyps - size greater than or equal to 10mm (yes/no)
    format the output as ***JSON output*** with the following schema for each colonoscopy procedure.  

    {'patient_name': '',
    'patient_NHI': '',
    
        'colonoscopy': [
                            {'date': '', 
                            'number of polyps': 0, 
                            },
        'histology': {
            {'adenoma': number of adenomas,
            'adenoma_size': 'largest adenoma size in mm',
            'high_grade_dysplasia_in the adenoma': 'yes' or 'no',
            'tubulovillous_or_villous_adenoma': 'yes' or 'no',
            'sessile_serrated_polyps': number of sessile serrated polyps,
            'sessile_serrated_polyp_size': 'size of largest sessile serrated polyp',
            'dysplasia_in_the_sessile_serrated_polyp': 'yes' or 'no',
            'hyperplastic_polyp_greater_or_equal_to_10mm_in_size: 'yes' or 'no',
        }
    }
    ]
}
    make sure the JSON is properly formatted and can be parsed by a standard JSON parser.
"""


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
    system_prompt = """
    you are a helpful medical assistant who is tasked with providing a detailed summary of only the pertinent details of the user input data that references recent colonoscopy procedures,
    the details from the procedure notes themselves, as well as the histological report from any polyps that were removed during that procedure.  Pertinent information that must be included
    in the summary are the number of polyps, the types of polyps (such as adenoma and whether the adenoma is tubulovillous or villous) , sessile serrated polyps, hyperplastic polyps as well as their sizes. 
    Regarding the procedure details, the significant findings include the BBPS score and where the scope was advanced to.  Regarding the polyps that are noted, please summarize and reconcile the information
    on the polyps that is contained within the procedure note as well as the histology report.  DO NOT make any clinical diagnoses or recommendations.
    """

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
    system_prompt = """
    
You are a helpful medical assistant who is tasked with making evidence based recommendations for follow up after a colonoscopy
        You will take the user query which includes medical information as well as the protocol that is pulled from the vector database, provide recommendations for follow up that meet the guidelines
        as stated in the documents that are provided.  If it is unclear what the recommendations should be or if the patient's data does not meet any of the criteria in the documents, simply output 
        'review by surgeon'. If there is uncertainty between two possible time intervals, choose the shorter one - for example if the choice is between 6 months or 12 months, choose 6 months.
        Provide a detailed explanation for your recommendation based on the protocols and data provided.  You do not need to generate any patient notes or documentation.  A recommendation
        based on the provided protocols and user query is all that is needed"""
    
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