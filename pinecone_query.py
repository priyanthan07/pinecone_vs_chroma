from openai import OpenAI
from dotenv import load_dotenv
import time
from pinecone import Pinecone

pc = Pinecone(api_key=) 
MODEL = "text-embedding-3-small"
load_dotenv()

client = OpenAI(api_key=)

def pinecone_query( query):
    try:
        index = pc.Index("knowledgebase")

        t = time.time()
        emb = client.embeddings.create(input=query, model=MODEL).data[0].embedding
        retrived_text = index.query(namespace="inbound-scenario" ,vector=[emb], top_k=4, include_metadata=True)
        e = time.time()
        query_time =  e-t
        
        output = [match['metadata']['text'] for match in retrived_text["matches"]]
        return "".join(output), query_time


    except Exception as e:
        raise e
    
if __name__ == "__main__":
    # for i in range(100):
    #     print(i+1)
    query = "How many investment professionals will be hired to support the new ESG portfolio?"
    result, q_time = pinecone_query(query)
    print(result)
    print(q_time)