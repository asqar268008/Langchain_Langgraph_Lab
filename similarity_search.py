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
index_name = "rag-lab-vectors"

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

documents = [
    Document(page_content="This is the first document."),
    Document(page_content="I am Asqar from Trichy"),
    Document(page_content="I am currently working on a project related to AI and machine learning."),
    Document(page_content="I have experience in Python, JavaScript, and data analysis."),
    Document(page_content="I am interested in exploring new technologies and staying updated with the latest trends in the tech industry."),
]

ids = vectorstore.add_documents(documents)
print(f"Added documents with IDs: {ids}")

query = "Which project is currently working?"

print(f"\nSearching for: {query}")

results = vectorstore.similarity_search(
    query,
    k=3,
)

print("\nSearch results:")

for doc in results:
    print(doc.page_content)