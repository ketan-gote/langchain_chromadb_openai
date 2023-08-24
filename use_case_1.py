import os
import chromadb
import openai
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import requests

persist_directory = "chroma_db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def read_docs():
    loader = DirectoryLoader("./documents")
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def save_chunks(chunks):
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()


def read_documents_and_index():
    docs = read_docs()
    print(f"Len={len(docs)} Content=", docs)

    chunk_docs = split_docs(docs)
    print("Chunk documents size=", len(chunk_docs))

    save_chunks(chunk_docs)


def search_docs(query):
    new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    matching_docs = new_db.similarity_search(query)

    return matching_docs


def integrate_chat_gpt_using_chain(docs, query):
    load_dotenv()

    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=0.0)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    # chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(query))

    answer = chain.run(input_documents=docs, question=query)

    return answer


def integrate_chat_gpt_with_own_kb(docs, query):
    load_dotenv()
    api_key = "Bearer "+os.getenv("OPENAI_API_KEY")
    url = 'https://api.openai.com/v1/chat/completions'

    knowledge_base = ""
    for match_docs in docs:
        knowledge_base = knowledge_base + match_docs.page_content

    knowledge_base = "Answer all my queries only on below context  \n {}".format(str(knowledge_base))
    print("knowledge_base=", knowledge_base)
    query_string = query + "  Make sure you answer on above context which I have provided"

    request = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": knowledge_base},
            {"role": "assistant",
             "content": "Sure, I'll provide answers to your queries based on the context you've provided."},
            {"role": "user", "content": query_string}
        ]
    }

    response = requests.post(url, json=request, headers={'Authorization': api_key})

    print("API call=", response)

    return response


def integrate_chat_gpt(query):
    load_dotenv()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please answer my queries"},
            {"role": "assistant", "content": "Ok"},
            {"role": "user", "content": query}
        ]
    )

    return response


def search_use_case():
    #query = "What are the emotional benefits of owning a pet?"
    #query = "How training pet is beneficial"
    #query = "What are the different kinds of pets people commonly own?"
    query = "Do you know Mike Tyson"

    match_docs = search_docs(query)
    print("Match docs =", match_docs)

    # chat_gpt_response1 = integrate_chat_gpt(query)
    chat_gpt_response2 = integrate_chat_gpt_with_own_kb(match_docs, query)

    chat_gpt_response2 = chat_gpt_response2.json()
    print("\nBased on ChatGPT KB ="+chat_gpt_response2["choices"][0]['message']['content'])


if __name__ == '__main__':
    read_documents_and_index()
    search_use_case()
