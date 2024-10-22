import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

def extract_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    
    return docs

if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    extracted_docs = extract_data(url)
    print(f"Extracted {len(extracted_docs)} document chunks")