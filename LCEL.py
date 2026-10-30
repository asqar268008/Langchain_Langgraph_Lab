import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words?"
)

llm = init_chat_model(
    model="qwen2.5-coder:3b",
    model_provider="ollama",
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke(
    {
        "topic": "langchain"
    }
)

print(result)