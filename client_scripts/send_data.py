#script to send data to the api endpoint

import os
import json
import requests
from dotenv import load_dotenv
import asyncio

load_dotenv()

data_path = os.getenv('DATA_PATH')
base_url = os.getenv('AZURE_APP_ENDPOINT')

report_file = os.path.join(data_path, 'sample_patient_report_6.txt')

with open(report_file, 'r', encoding = 'utf-8') as f:
    report = f.read()

api_url_rec = f'{base_url}/recommend'
api_url_summary = f'{base_url}/summarize'
api_url_json_summary = f'{base_url}/json_summary'

test_url_rec = 'http://localhost:8000/recommend'
test_url_summary = 'http://localhost:8000/summarize'
test_url_json_summary = 'http://localhost:8000/json_summary'


async def send_request(report_text: str, api_url: str):
    '''
    Sends a free text reort to the API endpoint and returns a recommendation based on the report and database contents
    '''

    data = {'user_query': report_text}
    headers = {
        'x-api-key': os.getenv('MY_API_KEY')
    }

    response = requests.post(api_url, json = data, headers = headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')
       
    

async def main():

    response_recommendation = await send_request(report, api_url_rec)
    json_summary = await send_request(report, api_url_json_summary)
    summary = await send_request(report, api_url_summary) #we just want summary in json format because it returns both a text summary and a json summary
    
    print(f'Summary Response: {summary}\n')
    print(f'JSON Summary Response: {json_summary}\n')
    print(f'Recommendation Response: {response_recommendation['recommendation']}')


if __name__ == '__main__':
    asyncio.run(main())

