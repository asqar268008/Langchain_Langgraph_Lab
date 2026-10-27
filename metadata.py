import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables.")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "metadata-lab"

if not pc.has_index(index_name):
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY"),
    output_dimensionality=1024,
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

document = [
    Document(
        page_content="My name is Sanjay viswanath",
        metadata = {"user_id": 1}
    ),

    Document(
        page_content="My name is Saranavan Sivakumar",
        metadata = {"user_id": 2}
    )
]

vectorstore.add_documents(document)

print("Document Inserted successfully!")

query = "What is my name?"

result = vectorstore.similarity_search(
    query=query,
    k=1,
    filter={"user_id":1}
)
for doc in result:
    print(doc.page_content)