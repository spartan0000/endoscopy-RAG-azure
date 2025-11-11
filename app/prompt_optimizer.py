#some prompt optimization code
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

import random

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime

  

from app.clients import chat_client, embedding_client

load_dotenv()

async def prompt_eval(prompt_variant: str, user_query: str, documents: List) -> str:
    document_contents = '\n'.join(documents)
    response = await chat_client.reponses.create(
        model = 'gpt-5-mini',
        input = [
            {'role': 'system',
             'content': prompt_variant},
            {'role': 'user',
             'content': f'User query: {user_query}\n\nDocuments:\n{document_contents}'}
        ]
    )
    output = response.output_text

    evaluation_prompt = f'''Evaluate the recommendation output for accuracy and completeness based on the user query and the documents provided
    Input: {user_query}
    Documents: {document_contents}
    Output: {output}
    Give a rating between 1 and 10 where 1 is poor and 10 is excellent.  Provide only the numeric score as output.
    '''

    score_response = await chat_client.responses.create(
        model = 'gpt-5-mini',
        input = [
            {'role': 'user',
             'content': evaluation_prompt}
        ]
    )
    try:
        score = float(score_response.output_text.strip())
    except:
        score = random.uniform(5,8) #default if parsing fails
    return {'prompt_variant': prompt_variant, 'score': score, 'output': output}

def mutate_prompt(prompt: str) -> str:
    mutations = [
        'Explicitly state what information or documents were used to make the recommendation',
        'Only use evidence from the retrieved documents to make the recommendation',
        'If evidence is weak or missing, state that explicitly in the recommendation and suggest review by a specialist',
        'If the recommendation is uncertain between two different options, state this explicitly and recommend the more conservative option',
        'If the information in the user query is not sufficient to make a recommendation, state this explicitly and suggest review by a specialist',
        'Avoid redundant or vague language or conclusions in the recommendation',
            
        ]
    addition = random.choice(mutations)
    mutated_prompt = f'{prompt} \n Optimization additon: {addition}'
    return mutated_prompt

