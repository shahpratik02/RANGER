#!/usr/bin/env python3
"""
Semantic Retriever for RepoBench Baseline

This script implements a semantic search baseline for code retrieval. It works by:
1. Connecting to a Neo4j graph of a repository.
2. Extracting code from Function, Method, and GlobalVariable nodes.
3. Generating embeddings for the code snippets.
4. Storing these embeddings in a ChromaDB vector store.
5. Retrieving the top-k most similar code snippets for a given query.
"""

import os
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_neo4j import Neo4jGraph
from src.utils.config import CONFIG

# Use config values
EMBEDDING_MODEL = CONFIG['models']['embedding']
CHROMA_CLIENT = chromadb.Client()
COLLECTION_NAME = CONFIG['chromadb']['collection_name']

# Global model variables
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading {EMBEDDING_MODEL} model...")
        # Load mixedbread-ai/mxbai-embed-large-v1 with half precision for memory efficiency
        # model_kwargs = {"torch_dtype": torch.float16}
        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
    return _model

def encode_text(texts):
    model = get_model()
    if isinstance(texts, str):
        texts = [texts]
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Simple embedding with mixedbread-ai/mxbai-embed-large-v1
    embeddings = model.encode(
        texts, 
        batch_size=32,  # Larger batch size for mixedbread model
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Convert to list format for compatibility
    return [emb for emb in embeddings]


def build_index_from_graph(repo_name, collection, neo4j_config):
    """
    Builds a vector index from the code snippets in the Neo4j graph.
    - Fetches code from Function, Method, and GlobalVariable nodes.
    - Splits large code snippets.
    - Generates and stores embeddings in ChromaDB.
    """
    print(f"Building index for repository: {repo_name}")

    # 1. Connect to Neo4j
    try:
        graph = Neo4jGraph(
            url=neo4j_config["url"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
        )
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return False

    # 2. Fetch nodes
    query = """
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS*]-(c)
    WHERE c:Function OR c:GlobalVariable
    RETURN id(c) AS id, c.name AS name, c.signature AS signature, c.description AS description, c.member_descriptions as member_descriptions, labels(c) AS type, c.module_name AS module, null AS class_name

    UNION

    MATCH (r:Repo {name: $repo_name})-[:CONTAINS*]-(c)-[:HAS_METHOD*]-(m:Method)
    RETURN id(m) AS id, m.name AS name, m.signature AS signature, m.description AS description, m.member_descriptions as member_descriptions, labels(m) AS type, m.module_name AS module, c.name AS class_name
    """
    try:
        nodes = graph.query(query, params={"repo_name": repo_name})
        print(f"  Found {len(nodes)} nodes to index.")
    except Exception as e:
        print(f"  Error querying graph: {e}")
        graph.close()
        return False

    # 3. Process and embed nodes
    documents = []
    metadatas = []
    ids = []

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=512, chunk_overlap=50
    )

    for node in nodes:
        description = node.get("description")
        if not description or not description.strip():
            continue

        node_type_list = node.get("type", [])
        member_descriptions = node.get("member_descriptions", "")
        description = description + "\n" + member_descriptions
        # Split text if it's too large
        code_chunks = text_splitter.split_text(description)

        for i, chunk in enumerate(code_chunks):
            metadata = {
                "name": node.get("name") or "",
                "type": ",".join(node_type_list),
                "module_name": node.get("module") or "",
                "repo_name": repo_name ,
                "original_node_id": node.get("id") or "",
                "class_name": node.get("class_name") or "",
            }
            if "GlobalVariable" not in node_type_list:
                metadata["signature"] = node.get("signature")

            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{node.get('id')}_{i}")

    if not documents:
        print("  No documents to index.")
        graph.close()
        return True  # It's not a failure if there's nothing to index

    # Generate embeddings
    print(f"  Generating embeddings for {len(documents)} document chunks...")
    embeddings = encode_text(documents)

    # 4. Store in ChromaDB
    print(
        f"  Storing {len(documents)} embeddings in ChromaDB collection '{collection.name}'..."
    )
    try:
        # Batch the insertions to avoid ChromaDB max batch size limit
        batch_size = 2000  # Safe batch size well below the limit
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            current_batch = (i // batch_size) + 1
            print(f"    Processing batch {current_batch}/{total_batches} ({len(batch_documents)} items)...")
            
            collection.add(
                embeddings=[emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings],
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        
        print("  ✓ Index built successfully.")
        return True
    except Exception as e:
        print(f"  Error adding to ChromaDB: {e}")
        return False
    finally:
        graph.close()


def retrieve_from_index(query, collection, top_k=5):
    """
    Retrieves the top_k most similar code snippets from the vector index.
    """
    if collection.count() == 0:
        print("  Warning: ChromaDB collection is empty. No retrieval possible.")
        return []

    # 1. Generate embedding for the query
    query_embedding = encode_text([query])[0].tolist()

    # 3. Query ChromaDB
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

    # 4. Format results
    formatted_results = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            formatted_results.append(
                {
                    "code": doc,
                    "name": metadata.get("name"),
                    "signature": metadata.get("signature"),
                    "module_name": metadata.get("module"),
                    "type": metadata.get("type"),
                    "class_name": metadata.get("class_name"),
                    "distance": results["distances"][0][i],
                }
            )

    return formatted_results


def main(question: str, repo_name: str, neo4j_config=None):
    """
    Main function to be called by the evaluation script.
    - Ensures the index is built for the repository.
    - Performs retrieval.
    """
    if neo4j_config is None:
        # Use config values as default
        neo4j_config = CONFIG['neo4j']

    # Get or create the ChromaDB collection
    collection = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)

    # Check if this repo is already indexed
    # A simple check: query for one item from the repo.
    existing_docs = collection.get(where={"repo_name": repo_name}, limit=1)
    if not existing_docs or not existing_docs["ids"]:
        print(f"Index for '{repo_name}' not found. Building it now.")
        # Clear collection before building new index to avoid mixing repos
        CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
        collection = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        build_index_from_graph(repo_name, collection, neo4j_config)

    # Perform retrieval
    retrieved_results = retrieve_from_index(question, collection, top_k=10)

    # The evaluation script expects a certain format.
    # The 'retirever_v2.py' script returns a list of dictionaries with 'signature' and 'code'.
    # We will mimic that here. The distance is extra info.
    final_results = []
    for res in retrieved_results:
        final_results.append(
            {
                "signature": res.get("signature") or res.get("name", ""),
                "code": res.get("code"),
                "name": res.get("name"),
                "distance": res.get("distance"),
                "class_name": res.get("class_name"),
                "module_name": res.get("module_name"),
            }
        )

    return final_results


if __name__ == "__main__":
    # This example assumes a Neo4j instance is running and contains data
    # for the specified repository.
    # IMPORTANT: Replace "test/repo" with a real repo_name that exists in your graph.
    # NOTE: Now using mixedbread-ai/mxbai-embed-large-v1 for improved embedding quality
    test_repo = "test/repo"  # Replace with a real repo name from your graph
    test_query = "def example_function(a, b):"

    print(f"--- Running Semantic Retriever Test for repo: {test_repo} ---")
    print(f"--- Using {EMBEDDING_MODEL} model ---")

    # The main function handles index creation internally
    results = main(question=test_query, repo_name=test_repo)

    print("\n--- Retrieval Results ---")
    if results:
        for i, res in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"  Signature: {res['signature']}")
            print(f"  Distance: {res['distance']:.4f}")
            print(f"  Code Snippet:\n---\n{res['code']}\n---")
    else:
        print("No results retrieved.")
