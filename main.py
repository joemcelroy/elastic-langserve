from elastic_rag_workplace import chain

if __name__ == "__main__":
    questions = [
        "What is the nasa sales team?",
        "What is our work from home policy?",
        "Does the company own my personal project?",
        "What job openings do we have?",
        "How does compensation work?",
    ]

    response = chain.invoke(questions[1])
    print(response)
