## initializing pinecone serverless

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from pinecone import Pinecone, ServerlessSpec 
from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()

client = OpenAI()
pc = Pinecone(api_key="1662dd38-7a38-4964-a176-89641d44c4f4")  

class Pinecone_Serverless:
    def __init__(self):
        self.MODEL = "text-embedding-3-small"
        self.index_name = "knowledgebase"

    def data_preprocess(self):
        try:
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
            texts = [p.strip() for p in knowledge_base]
            texts = [text for text in texts if text]

            character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=120, chunk_overlap=10)
            character_split_texts = character_splitter.split_text('\n'.join(texts))
            token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

            token_split_texts = []
            for text in character_split_texts:
                token_split_texts += token_splitter.split_text(text)

            return token_split_texts
        
        except Exception as e:
            raise e
    
    def init_pinecone(self):
        try:  
            spec = ServerlessSpec(cloud='aws', region='us-east-1')    
 
            pc.create_index(  
                self.index_name,  
                dimension=1536,  # dimensionality of text-embedding-ada-002  
                metric='dotproduct',  
                spec=spec  
            )
            
        except Exception as e:
            raise e
        
    def upload_embeddings(self):
        try:
            self.init_pinecone()
            token_split_texts = self.data_preprocess()
            index = pc.Index(self.index_name)

            count = 0  # we'll use the count to create unique IDs
            batch_size = 4  # process everything in batches of 32
            for i in range(0, len(token_split_texts), batch_size):
                # set end position of batch
                i_end = min(i+batch_size, len(token_split_texts))
                # get batch of lines and IDs
                lines_batch = token_split_texts[i: i+batch_size]
                ids_batch = [str(n) for n in range(i, i_end)]

                # create embeddings
                res = client.embeddings.create(input=lines_batch, model=self.MODEL)
                embeds = [record.embedding for record in res.data]

                # prep metadata and upsert batch
                meta = [{'text': line} for line in lines_batch]

                to_upsert = zip(ids_batch, embeds, meta)
                # upsert to Pinecone
                index.upsert(vectors=list(to_upsert), namespace="inbound-scenario")
                print(index.describe_index_stats())

            return {"status":"pinecone DB created successfully"}
        except Exception as e:
            raise e