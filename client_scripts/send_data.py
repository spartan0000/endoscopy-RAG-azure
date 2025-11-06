#script to send data to the api endpoint

import os
import json
import requests
from dotenv import load_dotenv
import asyncio

load_dotenv()

data_path = os.getenv('DATA_PATH')

report_file = os.path.join(data_path, 'sample_patient_report_1.txt')

with open(report_file, 'r', encoding = 'utf-8') as f:
    report = f.read()

api_url_rec = 'http://127.0.0.1:8000/recommend'
api_url_summary = 'http://127.0.0.1:8000/summarize'


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
    
    

async def main():

    response_recommendation = await send_request(report, api_url_rec)
    summary = await send_request(report, api_url_summary) #we just want summary in json format because it returns both a text summary and a json summary
    
    print(f'Summary Response: {summary}')
    print(f'Recommendation Response: {response_recommendation['recommendation']}')


if __name__ == '__main__':
    asyncio.run(main())

