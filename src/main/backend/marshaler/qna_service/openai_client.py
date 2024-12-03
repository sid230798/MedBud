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
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain.schema.runnable import RunnableMap
import re

import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = "/Users/yellow_flash/Git_Projects/BioMed/MedicalQnA_App/Med-QnA-App/.cache/transformers"
sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

def run_CoT_gpt(messages, medical_history, model="gpt-3.5-turbo-instruct"):
    load_dotenv()
    client = openai.OpenAI()
    conversation_history = "\n".join([message["content"] for message in messages])
    if medical_history != "":
        conversation_history += "\nWhile answering, also consider this as my medical history:\n" + medical_history
    
    answer = client.completions.create(
        model=model,
        prompt="Answer the last question of this conversation and explain the process step by step to arrive at a conclusion: " + conversation_history,
        max_tokens=500
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

def extract_paths(text):
    pattern = r"(Path\d+\s*:)"
    parts = re.split(pattern, text)
    
    # Combine the split parts into complete sections and remove "PathX" labels
    sections = []
    for i in range(1, len(parts), 2):  # Skip the text before the first "Path"
        section_content = parts[i + 1].strip()  # Ignore the Path label
        sections.append(section_content)
    
    return sections

def get_fact_candidates_similarity(fact, candidates):
    fact_embeddings = sentence_embedding_model.encode([fact])[0]
    candidates_embeddings = sentence_embedding_model.encode(candidates)
    sim_list = [0] * len(candidates)
    for index in range(len(candidates)):
        sim_list[index] = cos_sim(fact_embeddings, candidates_embeddings[index])
    return sim_list

def run_retrieval_CoT_pipeline(messages, medical_history, model="gpt-3.5-turbo-instruct", dataset="pubmedqa"):
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
    
    retriever_chain = (retriever | format_docs)
    facts = format_docs(retriever_chain.invoke(conversation_history))


    template = """ 
            Conversation: {question}
            Provide three reasoning paths for answering the question consicely, each considering different possible explanations. 
            For each path:
            - State the assumption.
            - Explain the reasoning process.
            - Indicate how it connects with the medical facts you know.
            - Conclusion for the answer

            Output Format:
            Path1 : ...
            Path2 : ...
            and so on....
    """

    # Define the RAG pipeline
    llm = OpenAI(model_name=model, openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template(template)
    chain = ({"question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser())
    
    # Run the RAG pipeline
    response = chain.invoke(conversation_history)
    if response == '':
        return "OpenAI API temporarily down :( Please try again in a while."
    
    paths = extract_paths(clean_response(response.strip()))
    sim_list = get_fact_candidates_similarity(facts, paths)

    best_path = paths[sim_list.index(max(sim_list))]
    template = """
        Additional context: {facts}
        Reasoning path: {best_path}
        conversation: {conversation}
        Use this reasoning path and additional context to output a brief answer for the last question in conversation.
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (RunnableMap({ "facts": RunnablePassthrough(),  # Pass as is
                "best_path": RunnablePassthrough(),  # Pass as is
                "conversation": RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser())
    
    
    # Run the RAG pipeline
    response = chain.invoke({
        "facts": facts,
        "best_path": best_path,
        "conversation": conversation_history
        })

    return clean_response(response.strip()) if response!= '' else "OpenAI API temporarily down :( Please try again in a while."