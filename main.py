from elastic_rag_workplace import chain
from langchain.memory import ElasticsearchChatMessageHistory

message_history = ElasticsearchChatMessageHistory(
    es_url="http://localhost:9200", index="workplace-msg-history", session_id="test"
)

if __name__ == "__main__":
    questions = [
        "What is the nasa sales team?",
        "What is our work from home policy?",
        "Does the company own my personal project?",
        "How does compensation work?",
    ]

    response = chain.invoke(
        {
            "question": questions[0],
            "chat_history": [],
        }
    )
    print(response)
    message_history.add_user_message(questions[0])
    message_history.add_ai_message(response)

    follow_up_question = "What are their objectives?"

    response = chain.invoke(
        {
            "question": follow_up_question,
            "chat_history": [
                "What is the nasa sales team?",
                "The sales team of NASA consists of Laura Martinez, the Area Vice-President of North America, and Gary Johnson, the Area Vice-President of South America. (Sales Organization Overview)",
            ],
        }
    )
    message_history.add_user_message(follow_up_question)
    message_history.add_ai_message(response)
    print(response)
