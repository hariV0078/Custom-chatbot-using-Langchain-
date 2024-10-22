from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

if __name__ == "__main__":
    from data_extraction import extract_data
    
    url = "https://brainlox.com/courses/category/technical"
    docs = extract_data(url)
    vectorstore = create_embeddings(docs)
    
    # Save the vector store
    vectorstore.save_local("faiss_index")