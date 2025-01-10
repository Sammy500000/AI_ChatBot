from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define path for the Chroma database
CHROMA_PATH = "chroma"

# Prompt template for generating responses
PROMPT_TEMPLATE = """
You are a highly intelligent and expert assistant. Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def create_instructor_embeddings():
    """
    Create embeddings using Ollama's embedding model.

    Returns:
        OllamaEmbeddings: Embedding function.
    """
    return OllamaEmbeddings(model="nomic-embed-text")

def query_database(query_text):
    """
    Query the Chroma database and generate a response using the LLM.

    Args:
        query_text (str): The user query.

    Returns:
        tuple: Response text and sources.
    """
    embedding_function = create_instructor_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=10)

    print(f"Raw results: {results}")

    if len(results) == 0:
        print("No results found.")
        return "Unable to find matching results.", []

    if results[0][1] < 0.4:
        print("Results found, but relevance score is below threshold.")
        return "Unable to find matching results.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    model = ChatOllama(model="tinyllama")
    llm_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template(PROMPT_TEMPLATE))
    response_text = llm_chain.run({"context": context_text, "question": query_text})

    sources = [doc.metadata.get("source", "No source provided") for doc, _score in results]
    return response_text, sources
