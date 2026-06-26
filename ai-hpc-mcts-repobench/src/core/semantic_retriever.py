#!/usr/bin/env python3
"""
Semantic Retriever for RepoBench Baseline

This script implements a semantic search baseline for code retrieval. It works by:
1. Connecting to a Neo4j graph of a repository.
2. Extracting code from Function, Method, and GlobalVariable nodes.
3. Generating embeddings for the code snippets using Qwen3-Embedding-8B.
4. Storing these embeddings in a ChromaDB vector store.
5. Retrieving the top-k most similar code snippets for a given query.
"""

import os
import chromadb
import torch
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from transformers import AutoTokenizer, AutoModel
from langchain_neo4j import Neo4jGraph

# Import configuration
from src.utils.config import CONFIG

# --- Configuration from config file ---
EMBEDDING_MODEL = CONFIG['models']['embedding']['name']
BATCH_SIZE = CONFIG['models']['embedding']['batch_size']
CONTEXT_LENGTH = CONFIG['models']['embedding']['context_length']

CHROMA_CLIENT = chromadb.Client()
COLLECTION_NAME = CONFIG['vector_db']['chroma']['collection_name']

# Global model variables
_model = None
_tokenizer = None

def last_token_pool(last_hidden_states, attention_mask):
    """Extract embeddings using last token pooling (for Qwen3)"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {EMBEDDING_MODEL} model...")
        _tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, 
            padding_side=CONFIG['models']['embedding']['padding_side']
        )
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=CONFIG['models']['embedding']['trust_remote_code']
        )
        _model.eval()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        _model.to(device)
    return _model, _tokenizer

def encode_text(texts):
    model, tokenizer = get_model()
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    device = next(model.parameters()).device
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=CONTEXT_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # Convert to numpy and add to results
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return embeddings

def build_index_from_graph(repo_name, collection, neo4j_config):
    """
    Builds a vector index from the code snippets in the Neo4j graph.
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
    MATCH (n)
    WHERE n:Function OR n:Method OR n:GlobalVariable
    RETURN id(n) AS id, n.name AS name, n.signature AS signature, n.code AS code, labels(n) AS type, n.module_name AS module
    """
    try:
        nodes = graph.query(query)
        print(f"  Found {len(nodes)} nodes to index.")
    except Exception as e:
        print(f"  Error querying graph: {e}")
        graph.close()
        return False

    # 3. Process and embed nodes
    documents = []
    metadatas = []
    ids = []

    # Initialize text splitter with smaller chunk size for better context fit
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2048, chunk_overlap=100
    )

    for node in nodes:
        code = node.get("code")
        if not code or not code.strip():
            continue

        node_type_list = node.get("type", [])
        # Split code if it's too large
        code_chunks = text_splitter.split_text(code)

        for i, chunk in enumerate(code_chunks):
            metadata = {
                "name": node.get("name") or "",
                "type": ",".join(node_type_list),
                "module": node.get("module") or "",
                "repo_name": repo_name,
                "original_node_id": node.get("id") or "",
            }
            if "GlobalVariable" not in node_type_list:
                metadata["signature"] = node.get("signature")

            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{node.get('id')}_{i}")

    if not documents:
        print("  No documents to index.")
        graph.close()
        return True

    # Generate embeddings
    print(f"  Generating embeddings for {len(documents)} document chunks...")
    embeddings = encode_text(documents)

    # 4. Store in ChromaDB
    print(f"  Storing {len(documents)} embeddings in ChromaDB collection '{collection.name}'...")
    try:
        # Batch the insertions
        batch_size = 2000
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

    # 2. Query ChromaDB
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

    # 3. Format results
    formatted_results = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            formatted_results.append(
                {
                    "code": doc,
                    "name": metadata.get("name"),
                    "signature": metadata.get("signature"),
                    "module": metadata.get("module"),
                    "type": metadata.get("type"),
                    "distance": results["distances"][0][i],
                }
            )

    return formatted_results

def main(question: str, repo_name: str, neo4j_config=None):
    """
    Main function to be called by the evaluation script.
    """
    print(f"Semantic Retrieval for question: {question[:100]}...")
    
    if neo4j_config is None:
        # Use configuration from config file
        neo4j_config = CONFIG['database']['neo4j']

    # Get or create the ChromaDB collection
    collection = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
    print("##### Using Model: ", EMBEDDING_MODEL)
    # Check if this repo is already indexed
    existing_docs = collection.get(where={"repo_name": repo_name}, limit=1)
    if not existing_docs or not existing_docs["ids"]:
        print(f"Index for '{repo_name}' not found. Building it now.")
        # Clear collection before building new index
        CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
        collection = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        build_index_from_graph(repo_name, collection, neo4j_config)

    # Perform retrieval
    retrieved_results = retrieve_from_index(question, collection, top_k=5)

    # Format results for evaluation script
    final_results = []
    for res in retrieved_results:
        final_results.append(
            {
                "signature": res.get("signature") or res.get("name", ""),
                "code": res.get("code"),
                "name": res.get("name"),
                "distance": res.get("distance"),
            }
        )

    return final_results

# Example usage
if __name__ == "__main__":
    test_repo = "test/repo"  # Replace with a real repo name from your graph
    test_query = "def example_function(a, b):"

    print(f"--- Running Semantic Retriever Test for repo: {test_repo} ---")
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
