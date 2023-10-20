from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableMap
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.memory import ElasticsearchChatMessageHistory
from langchain.schema import format_document
from typing import Tuple, List
from operator import itemgetter

message_history = ElasticsearchChatMessageHistory(
    es_url="http://localhost:9200", index="workplace-msg-history", session_id="test"
)

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

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If there is no chat history, just rephrase the question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
Use the following passages to answer the user's question.
Each passage has a SOURCE which is the title of the document. When answering, cite source name of the passages you are answering from below the answer in a unique bullet point list.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

----
{context}
----
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

document_prompt = PromptTemplate.from_template(
    template="""
---
NAME: {name}
PASSAGE:
{page_content}
---
""",
)


def _combine_documents(
    docs, document_prompt=document_prompt, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

chain = _inputs | _context | prompt | llm | StrOutputParser()
