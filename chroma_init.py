import os
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.HttpClient(host="54.226.228.250", port=8000)
# chroma_client = chromadb.Client()

# chroma_collection = chroma_client.get_or_create_collection("test-knowledgebase", embedding_function=embedding_function)



def chroma_hosted():
    
    chroma_collection = chroma_client.get_or_create_collection("test-knowledgebase", embedding_function=embedding_function)
    knowledge_base = '''
                Name: Phillipe BensimonPosition: Portfolio Manager, overseeing ESG investment strategies at your employer, 
                Point27 Asset Management Age: 45 years old, with 20+ years of experience in the investment industry Point27 Asset 
                Management company information * Hedge fund that invests in global security markets* $3 billion in assets under 
                management, and growing quickly* Point27 is a multi-strategy fund, pursuing investments in all asset classes 
                including equities, fixed income, currencies, commodities and private equity. * Point27 just raised an additional $1 
                billion for an ESG focused investment fund, which Phillipe Bensimon will now run. * The investment team is expected to 
                grow significantly thanks to this new capital, with at least another 10 investment professionals to be hired to support 
                the new ESG portfolio. You are open to discussing the purchase of additional Bloomberg terminal licenses for these new 
                employees, so long as the Bloomberg sales person asks you directly. only accessible if the user asks about a growth 
                opportunity
            '''
    lines = knowledge_base.split("\n")

    texts = [p.strip() for p in lines]
    texts = [text for text in texts if text]
    
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=100, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n'.join(texts))
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
        
    ids = [str(i) for i in range(len(token_split_texts))]  
        
    chroma_collection.add(ids=ids, documents=token_split_texts)
    
    return {"status":"chroma DB created successfully"}

if __name__ == "__main__":
    print(chroma_hosted())

