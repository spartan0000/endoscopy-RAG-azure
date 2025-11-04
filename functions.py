import openai
from openai import OpenAI, AzureOpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv
import langchain

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
from datetime import datetime


load_dotenv()

