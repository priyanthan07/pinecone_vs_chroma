import streamlit as st
from pinecone_init import Pinecone_Serverless
from pinecone_query import pinecone_query
from pinecone import Pinecone
from chroma_init import chroma_hosted, chroma_client
from chroma_query import Chroma_query
from fastapi import APIRouter\

pc = Pinecone(api_key="1662dd38-7a38-4964-a176-89641d44c4f4") 
index_name = "test-knowledgebase"

router = APIRouter()

@router.get("/pinecone")
async def InitPinecone():
    pc = Pinecone(api_key="1662dd38-7a38-4964-a176-89641d44c4f4") 
    index_name = "knowledgebase"
    if index_name not in pc.list_indexes().names():
        pc = Pinecone_Serverless()
        status = pc.upload_embeddings()
        return status

@router.get("/chroma")
async def InitChroma():
    index_name = "test-knowledgebase"
    collections = [collection.name for collection in chroma_client.list_collections()]
    print(collections)
    if index_name not in collections:
        status = chroma_hosted()
        return status
    
@router.post("/delete-knowledge-base") 
async def delete_knowledge_base(name):
    chroma_client.delete_collection(name)
    return "deleted"


@router.post("/pinecone_chat")
async def pineconeChat(userMessage):
    p_result,  query_time = pinecone_query(userMessage)
    return {"result" : p_result , "query_time" : query_time}

@router.post("/chroma_chat")
async def chromaChat(userMessage):
    c_result, query_time = Chroma_query(userMessage)
    return {"result" : c_result , "query_time" : query_time}

  