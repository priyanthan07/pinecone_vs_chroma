import time
import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.HttpClient(host="54.226.228.250", port=8000)
chroma_collection = chroma_client.get_collection("test-knowledgebase")

def Chroma_query(query):
    
    s1 = time.time()
    results = chroma_collection.query(query_texts=[query], n_results=4)
    e1 = time.time()
    query_time =  e1-s1
    
    retrieved_documents = results['documents'][0]
    source_knowledge = "\n".join([result for result in retrieved_documents])

    return source_knowledge, query_time


if __name__ == "__main__":
    query = "How many investment professionals will be hired to support the new ESG portfolio?"
    result, q_time = Chroma_query(query)
    print(result)
    print(q_time)