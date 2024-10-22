from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def create_chatbot():
    # Load the saved vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Create a conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Load Hugging Face model and tokenizer
    model_name = "gpt2"  # You can change this to a larger model if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token_id=50256)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a text generation pipeline with truncation
    text_generation = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=3000,  # Increase this value to handle longer inputs
        max_new_tokens=1000,  # Set the maximum number of new tokens to generate
        truncation=True,  # Explicitly enable truncation
        device=torch.device("cpu")  # Set the device
    )

    # Create a HuggingFacePipeline language model
    llm = HuggingFacePipeline(pipeline=text_generation)

    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa

def get_response(qa, query):
    print(f"Received query: {query}")
    print(f"Query length: {len(query)}")

    try:
        # Debugging: Check if query is too long
        if len(query) > 1024:  # Adjust based on model's capacity
            return "The input query is too long. Please provide a shorter query."

        print("Retrieving relevant documents...")

        # Retrieve documents related to the query
        relevant_docs = qa.retriever.get_relevant_documents(query)

        # Debug: Print out the number of documents retrieved
        print(f"Number of documents retrieved: {len(relevant_docs)}")

        # Handle the case where no documents are retrieved
        if len(relevant_docs) == 0:
            return "Sorry, I couldn't find any relevant documents for your query."

        # Use invoke instead of __call__
        result = qa.invoke({"question": query})

        # Check if result contains expected data
        if not result or "answer" not in result:
            print("No answer found in the result.")
            return "Sorry, I couldn't generate an answer for your query."

        print(f"Generated result: {result}")
        return result["answer"]

    except IndexError as e:
        print(f"IndexError encountered: {e}")
        return "Sorry, an internal error occurred due to an index out of range."
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Sorry, something went wrong."

if __name__ == "__main__":
    chatbot = create_chatbot()
    while True:
        query = input("User: ")
        if query.lower() in ['quit', 'exit', 'bye']:
            break
        response = get_response(chatbot, query)
        print(f"Chatbot: {response}")
