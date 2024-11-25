import os
import pickle

import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
import re


def clean_response(response):
    cleaned_response = re.sub(r'^\W+', '', response)
    return cleaned_response

def request_gpt_no_rag(messages, medical_history, model):
    load_dotenv()
    client = openai.OpenAI()
    conversation_history = "\n".join([message["content"] for message in messages])
    if medical_history != "":
        conversation_history += "\nWhile answering, also consider this as my medical history:\n" + medical_history

    answer = client.completions.create(
        model=model,
        prompt="Answer the last question of this conversation briefly: " + conversation_history,
        max_tokens=200
    )
    return clean_response(answer.choices[0].text.strip()) if answer!= '' else "OpenAI API temporarily down :( Please try again in a while."


def run_rag_pipeline(messages, medical_history, model="gpt-3.5-turbo-instruct", dataset="nfcorpus"):
    load_dotenv()
    conversation_history = "\n".join([message["content"] for message in messages])
    if medical_history != "":
        conversation_history += "\nWhile answering, also consider this as my medical history:\n" + medical_history

    # Load index from file
    loaded_faiss_vs = FAISS.load_local(
        # folder_path=f"src/main/backend/qna_service/datastore/vectordb/faiss/{dataset}/", # Uncomment for dev
        folder_path=f"./qna_service/datastore/vectordb/faiss/{dataset.lower()}/",   # Comment for dev
        embeddings=OpenAIEmbeddings())
    retriever = loaded_faiss_vs.as_retriever(search_kwargs={"k": 5})

    # Define the RAG pipeline
    llm = OpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"))

    template = """Answer the last question of the conversation, given this additional context: {context}
    Conversation: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    # docs_file_path = f"src/main/backend/qna_service/datastore/dataset/{dataset}/documents.pkl" # Uncomment for dev
    docs_file_path = f"./qna_service/datastore/dataset/{dataset.lower()}/documents.pkl" # Comment for dev
    with open(docs_file_path, "rb") as file:
        docs = pickle.load(file)

    def format_docs(_docs):
        ls = []
        for doc in _docs:
            if doc.page_content in docs:
                ls.append(docs[doc.page_content]["text"][:800])
        return ls

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser())

    # Run the RAG pipeline
    response = chain.invoke(conversation_history)

    return clean_response(response.strip()) if response!= '' else "OpenAI API temporarily down :( Please try again in a while."
