import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# NOTE: this script is meant to be run from the root of the repository
if __name__ == "__main__":
    # initialize chroma client
    chroma_client = chromadb.PersistentClient(".chroma")

    # initialize embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    # create a collection
    collection = chroma_client.create_collection(name="biodex-reaction-terms-test", embedding_function=openai_ef)

    # load reaction terms
    reaction_terms = []
    with open("testdata/reaction_terms.txt") as f:
        for line in f:
            reaction_terms.append(line.strip())

    # insert documents
    collection.add(
        documents=reaction_terms[:10],
        ids=[f"id{idx + 1}" for idx in range(len(reaction_terms[:10]))]
    )
