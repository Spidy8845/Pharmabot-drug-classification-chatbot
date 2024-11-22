# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:30:31 2024

@author: HP
"""
import streamlit as st
import os 
from langchain_community.llms import GooglePalm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader


st.title("Pharma ChatBot")
btn = st.button("Answer")

if btn:
    pass

def main():
        
    api_key = 'AIzaSyCH12u9reoFaowdRra92MymIZuU9kDLsjg'
    
    llm = GooglePalm(google_api_key=api_key,temperature=0)
    
    
    #emmbedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb_path1 = r"F:\360 digiTMG\ML Project\project dataset\db\Spider_db"
    file_path1 = r"F:\360 digiTMG\ML Project\project dataset\KMCOST_DATA\data_5000_!.csv"
    
    def create_vector_db():
        loader = CSVLoader(file_path = file_path1)
        documents = loader.load()
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 1000 )
        texts = text_splitter.split_documents(documents)
        
        vectordb1 = FAISS.from_documents(documents=texts, embedding = embeddings)
        vectordb1.save_local(vectordb_path1)
    
    
    def get_qa_chain():
        #vectordb1 = FAISS.load_local(vectordb_path1, embeddings)
        # Load local FAISS index with allow_dangerous_deserialization=True
        vectordb1 = FAISS.load_local(vectordb_path1, embeddings, allow_dangerous_deserialization=True)
    
        prompt_template = """ Given the following context and a question, generate an answer
        based on the context only. In the answer try to provide as much text as possible from 'response' section in the source document
        without making. if the answer is not found in the context kindly state "It is a New Drug". Don't try to make up an answer .
    
        CONTEXT: {context}
        QUESTION: {question} """
    
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
        
        retriever = vectordb1.as_retriever()
    
        from langchain.chains import RetrievalQA
        chain1 = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=retriever,
                                            input_key='query',
                                            return_source_documents = False,
                                            chain_type_kwargs = {"prompt":PROMPT}
                                            )
        
        return chain1
    
    # while True:

    #     query = input('Input Prompt : ')
    #     if query.lower() == 'exit':
    #         print('Exiting...')
    #         break
    
    #     if query == '':
    #         continue
    
    #     result = chain1(query)
    #     print("Response:", result)

    


question = st.text_input("Query:")

if question:
    chain = get_qa_chain()
    response = chain(question)
    
    
    st.header("Answer:")
    st.write(response["result"])



if __name__ == "__main__":
    main()
    