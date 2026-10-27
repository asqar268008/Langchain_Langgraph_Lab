import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables.")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "rag"

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

vectorstore.add_documents(documents)

print("Document Inserted successfully!")

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k":1,
    }
)

model = init_chat_model(
    model = "gemini-3.5-flash",
    model_provider="google-genai",
    api_key = os.getenv("API_KEY")
)

@tool
def search(query: str) -> str:
    """
    search document in vector DB (pinecone)
    """
    docs = retriever.invoke(query)
    return "\n".join(
        doc.page_content for doc in docs
    )

agent = create_agent(
    model = model,
    tools = [search],
    system_prompt=(
        "use search tool when documents is needed"
    )
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "where is live in Asqar"
            }
        ]
    }
)

print("Response:", response["messages"][-1].content)