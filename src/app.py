from src.pipeline import build_pipeline, ask_question

if __name__ == "__main__":
    print("Building RAG system...")
    vs = build_pipeline("data/sample.txt")

    while True:
        query = input("\nAsk your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = ask_question(query, vs)
        print("\nAnswer:", answer)
