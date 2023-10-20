from elastic_rag_workplace import chain

if __name__ == "__main__":
    questions = [
        "What is the nasa sales team?",
        "What is our work from home policy?",
        "Does the company own my personal project?",
        "How does compensation work?",
    ]

    response = chain.invoke(
        {
            "question": questions[3],
            "chat_history": [],
        }
    )
    print(response)
