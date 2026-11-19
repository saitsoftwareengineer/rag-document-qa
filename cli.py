"""CLI: python cli.py "What happens when drift is detected?" """
import sys

from rag_qa.pipeline import RagPipeline

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How are models deployed in this system?"
    pipe = RagPipeline()
    pipe.ingest()
    result = pipe.ask(question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
