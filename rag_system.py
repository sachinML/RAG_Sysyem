# import openai
# import re
# import PyPDF2
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import spacy
# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize resources
# nlp = spacy.load('en_core_web_sm')
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # OpenAI API setup
# openai.api_key = os.getenv("openai_key")

# # Load and preprocess the PDF (NCERT)
# def extract_text_from_pdf(pdf_path):
#     reader = PyPDF2.PdfReader(pdf_path)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# pdf_text = extract_text_from_pdf("./iesc111.pdf")

# # Split text into chunks
# chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
# embeddings = model.encode(chunks)

# # Initialize FAISS vector database
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))

# # Function to process RAG queries
# def process_rag_query(query):
#     query_embedding = model.encode([query])
#     D, I = index.search(query_embedding, k=5)
#     relevant_docs = [chunks[i] for i in I[0]]
    
#     system_prompt = "You are an expert in reading document. given data is extracted from the pdf. \
#                     You need to answer as per the question asked by the user."
#     question = f"Answer the question '{query}' based on the given content."
#     content_str = "\n".join(relevant_docs)
    
#     # Call the OpenAI GPT model to generate the response
#     response = openai.ChatCompletion.create(
#         model='gpt-4o-mini',
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {
#                 "role": "user",
#                 "content": f"Content: {content_str}\nConclusion: {question}",
#             },
#         ],
#         temperature=0.0,
#         max_tokens=100,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#     )
    
#     return response.choices[0].message.content.strip()

# # Additional Agent functionalities
# def classify_query(query):
#     if re.search(r'\b(hello|hi|hey|how are you)\b', query, re.IGNORECASE):
#         return 'greeting'
#     elif re.search(r'weather', query, re.IGNORECASE):
#         return 'weather_lookup'
#     elif re.search(r'\d', query):
#         return 'math_calculation'
#     else:
#         return 'ncert_query'

# def evaluate_math_expression(query):
#     try:
#         return str(eval(query))
#     except:
#         return "Sorry, I can't calculate that."

# def get_weather_info(city):
#     api_key = "your_openweathermap_api_key"
#     weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
#     response = requests.get(weather_url)
#     if response.status_code == 200:
#         data = response.json()
#         weather = data['weather'][0]['description']
#         temp = round(data['main']['temp'] - 273.15, 2)
#         return f"The weather in {city} is {weather} with a temperature of {temp}°C."
#     else:
#         return "Sorry, I couldn't fetch the weather for that location."

# # Function to process Agent queries
# def process_agent_query(query):
#     action = classify_query(query)
    
#     if action == 'greeting':
#         return "Hello! How can I assist you today?"
    
#     elif action == 'weather_lookup':
#         city = query.split()[-1]  # Assume the last word is the city
#         return get_weather_info(city)
    
#     elif action == 'math_calculation':
#         return evaluate_math_expression(query)
    
#     elif action == 'ncert_query':
#         return process_rag_query(query)
    
#     else:
#         return "I'm not sure how to handle that query."
    

import openai
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize resources
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# OpenAI API setup
openai.api_key = os.getenv("openai_key")

# Sarvam API key and endpoint
sarvam_api_key = os.getenv("sarvam_api_key")
sarvam_tts_endpoint = "https://api.sarvam.com/tts"

# Load and preprocess the PDF (NCERT)
def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf("./iesc111.pdf")

# Split text into chunks
chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
embeddings = model.encode(chunks)

# Initialize FAISS vector database
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Function to process RAG queries
def process_rag_query(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)
    relevant_docs = [chunks[i] for i in I[0]]
    
    system_prompt = "You are an expert in reading document. given data is extracted from the pdf. \
                    You need to answer as per the question asked by the user."
    question = f"Answer the question '{query}' based on the given content."
    content_str = "\n".join(relevant_docs)
    
    # Call the OpenAI GPT model to generate the response
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Content: {content_str}\nConclusion: {question}",
            },
        ],
        temperature=0.0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    
    return response.choices[0].message.content.strip()

# Function to convert text response to speech using Sarvam API
def convert_text_to_speech(text):
    headers = {
        "Authorization": f"Bearer {sarvam_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice": "en-US",
        "format": "mp3"
    }
    response = requests.post(sarvam_tts_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        audio_url = response.json().get("audio_url")
        return audio_url
    else:
        return None

# Additional Agent functionalities
def classify_query(query):
    if re.search(r'\b(hello|hi|hey|how are you)\b', query, re.IGNORECASE):
        return 'greeting'
    elif re.search(r'weather', query, re.IGNORECASE):
        return 'weather_lookup'
    elif re.search(r'\d', query):
        return 'math_calculation'
    else:
        return 'ncert_query'

def evaluate_math_expression(query):
    try:
        return str(eval(query))
    except:
        return "Sorry, I can't calculate that."

def get_weather_info(city):
    api_key = "your_openweathermap_api_key"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(weather_url)
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temp = round(data['main']['temp'] - 273.15, 2)
        return f"The weather in {city} is {weather} with a temperature of {temp}°C."
    else:
        return "Sorry, I couldn't fetch the weather for that location."

# Function to process Agent queries with voice response
def process_agent_query(query):
    action = classify_query(query)
    
    if action == 'greeting':
        text_response = "Hello! How can I assist you today?"
    
    elif action == 'weather_lookup':
        city = query.split()[-1]
        text_response = get_weather_info(city)
    
    elif action == 'math_calculation':
        text_response = evaluate_math_expression(query)
    
    elif action == 'ncert_query':
        text_response = process_rag_query(query)
    
    else:
        text_response = "I'm not sure how to handle that query."
    
    # Convert text response to speech
    audio_url = convert_text_to_speech(text_response)
    
    return {"text_response": text_response, "audio_url": audio_url}

