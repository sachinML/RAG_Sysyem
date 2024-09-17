import streamlit as st
import requests

# Define the FastAPI backend URL
api_url = "http://localhost:8000/rag/"

# Streamlit UI
st.title("NCERT RAG System")

query = st.text_input("Enter your query related to NCERT content:")

if st.button("Submit"):
    if query:
        response = requests.post(api_url, json={"query": query})
        if response.status_code == 200:
            st.write("Response from RAG system:")
            st.write(response.json().get('response'))
        else:
            st.write("Error: Unable to retrieve response.")
    else:
        st.write("Please enter a query.")
