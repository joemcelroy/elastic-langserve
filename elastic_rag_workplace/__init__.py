from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore

# Setup connecting to Elasticsearch
vectorstore = ElasticsearchStore(
    embedding=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    ),
    es_url="http://localhost:9200",
    index_name="workplace-search-example",
)
retriever = vectorstore.as_retriever()

# Set up LLM to user
llm = ChatOpenAI(temperature=0)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
