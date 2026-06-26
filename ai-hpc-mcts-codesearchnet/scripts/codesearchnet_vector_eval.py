#!/usr/bin/env python3
"""
CodeSearchNet Vector Database Evaluation Script

This script:
1. Loads the entire CodeSearchNet corpus
2. Groups documents by repository
3. Creates vector databases for each repository with doc_id metadata
4. Evaluates retrieval by comparing doc_ids instead of function names
5. Uses Qwen3-8B embedding model
6. Properly cleans up vector databases after each repository
"""

import pickle
import math
import os
import ir_datasets
from collections import defaultdict, Counter
import chromadb
import torch
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from transformers import AutoTokenizer, AutoModel
import json
from src.utils.simple_config import CONFIG

# Configuration
EMBEDDING_MODEL = CONFIG['models']['qwen_embedding']  # Using Qwen3-8B embedding model as requested
CHROMA_CLIENT = chromadb.Client()

# Global model variables
_model = None
_tokenizer = None

def get_model():
    """Load and return the Qwen3-8B embedding model"""
    global _model, _tokenizer
    if _model is None:
        print("Loading Qwen embedding model...")
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side='left')
        _model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        _model.eval()
        if torch.cuda.is_available():
            _model.to("cuda")
    return _model, _tokenizer

def last_token_pool(last_hidden_states, attention_mask):
    """Last token pooling for Qwen models"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def encode_text(texts):
    """Generate embeddings for text using Qwen model"""
    model, tokenizer = get_model()
    if isinstance(texts, str):
        texts = [texts]
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    embeddings = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        max_length = 4096
        batch_size = 8  # Smaller batch size for stability
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            outputs = model(**batch_dict)
            embeddings_tensor = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
            
            # Convert to numpy and add to list
            for j in range(embeddings_tensor.shape[0]):
                embeddings.append(embeddings_tensor[j].cpu().numpy())
            
            # Clean up memory after each batch
            del batch_dict, outputs, embeddings_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return embeddings

def build_vector_index_from_docs(repo_name, docs_list, collection):
    """
    Build vector index from CodeSearchNet documents for a repository
    """
    print(f"Building vector index for repository: {repo_name}")
    print(f"Processing {len(docs_list)} documents...")
    
    documents = []
    metadatas = []
    ids = []
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=512, chunk_overlap=50
    )
    
    for doc in docs_list:
        code = doc.code
        if not code or not code.strip():
            continue
            
        # Split code if it's too large
        code_chunks = text_splitter.split_text(code)
        
        for i, chunk in enumerate(code_chunks):
            metadata = {
                "doc_id": doc.doc_id,  # Store doc_id in metadata for comparison
                "repo": doc.repo,
                "path": doc.path,
                "func_name": doc.func_name,
                "language": doc.language,
                "chunk_index": i
            }
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{doc.doc_id}_{i}")
    
    if not documents:
        print("  No documents to index.")
        return True
    
    # Generate embeddings
    print(f"  Generating embeddings for {len(documents)} document chunks...")
    embeddings = encode_text(documents)
    
    # Store in ChromaDB
    print(f"  Storing {len(documents)} embeddings in ChromaDB...")
    try:
        batch_size = 1000  # Safe batch size
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            current_batch = (i // batch_size) + 1
            print(f"    Processing batch {current_batch}/{total_batches}...")
            
            collection.add(
                embeddings=[emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings],
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        
        print("  ✓ Vector index built successfully.")
        return True
    except Exception as e:
        print(f"  Error adding to ChromaDB: {e}")
        return False

def retrieve_from_vector_index(query, collection, top_k=10):
    """
    Retrieve top_k most similar documents from vector index
    """
    if collection.count() == 0:
        print("  Warning: ChromaDB collection is empty.")
        return []
    
    # Generate embedding for query
    query_embedding = encode_text([query])[0].tolist()
    
    # Query ChromaDB
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []
    
    # Format results
    formatted_results = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            formatted_results.append({
                "code": doc,
                "doc_id": metadata.get("doc_id"),
                "func_name": metadata.get("func_name"),
                "path": metadata.get("path"),
                "distance": results["distances"][0][i],
            })
    
    return formatted_results

def evaluate_retrieval_vector(query_id, query_lookup, query_to_docs, collection):
    """
    Evaluate retrieval using vector database by comparing doc_ids
    """
    query = query_lookup[query_id][1]  # Get query text
    
    # Get results from vector retrieval
    vector_results = retrieve_from_vector_index(query, collection, top_k=10)
    
    # Get ground truth documents for this query
    ground_truth_docs = query_to_docs.get(query_id, [])
    
    # Create a mapping from doc_id to relevance scores
    doc_id_to_relevance = {}
    for doc_info in ground_truth_docs:
        doc_id = doc_info['doc_id']
        relevance = int(doc_info['relevance'])
        doc_id_to_relevance[doc_id] = relevance
    
    # Check if there are any relevant ground truth entries
    relevant_gt_count = sum(1 for rel in doc_id_to_relevance.values() if rel > 0)
    if relevant_gt_count == 0:
        print(f"Skipping query {query_id}: No relevant ground truth entries")
        return None
    
    # Extract doc_ids from retrieved results and create relevance vector
    retrieved_relevances = []
    matched_gt_items = set()  # Track which ground truth items have been matched
    
    for i in range(min(10, len(vector_results))):  # Only consider top 10
        result = vector_results[i]
        retrieved_doc_id = result.get('doc_id')
        
        # Get relevance score (0 if not found in ground truth)
        relevance = 0
        if retrieved_doc_id in doc_id_to_relevance and retrieved_doc_id not in matched_gt_items:
            relevance = doc_id_to_relevance[retrieved_doc_id]
            matched_gt_items.add(retrieved_doc_id)
        
        retrieved_relevances.append(relevance)
        print(f"  Retrieved doc_id: {retrieved_doc_id}, relevance: {relevance}")
    
    # Pad with zeros if we have less than 10 results
    while len(retrieved_relevances) < 10:
        retrieved_relevances.append(0)
    
    # Calculate NDCG@10
    def calculate_dcg(relevances):
        dcg = 0.0
        for i, rel in enumerate(relevances):
            if rel > 0:
                dcg += rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
        return dcg
    
    # DCG for retrieved results
    dcg = calculate_dcg(retrieved_relevances)
    
    # IDCG - ideal DCG (sort ground truth by relevance)
    ideal_relevances = sorted([int(doc_info['relevance']) for doc_info in ground_truth_docs], reverse=True)
    ideal_relevances = ideal_relevances[:10]  # Top 10
    while len(ideal_relevances) < 10:
        ideal_relevances.append(0)
    
    idcg = calculate_dcg(ideal_relevances)
    
    # NDCG@10
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0
    
    # Calculate Recall@10
    relevant_retrieved = sum(1 for rel in retrieved_relevances if rel > 0)
    total_relevant = sum(1 for doc_info in ground_truth_docs if int(doc_info['relevance']) > 0)
    recall_10 = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    return {
        'ndcg_10': ndcg_10,
        'recall_10': recall_10,
        'query_id': query_id,
        'total_relevant': total_relevant,
        'relevant_retrieved': relevant_retrieved
    }

def deduplicate_qrels(qrels_list):
    """
    Process qrels list to get unique query_id, doc_id pairs with mode relevance scores
    """
    pairs_to_relevances = defaultdict(list)
    
    for i in range(len(qrels_list)):
        query_id = qrels_list[i][0]
        doc_id = qrels_list[i][1]
        relevance = qrels_list[i][2]
        note = qrels_list[i][3] if len(qrels_list[i]) > 3 else ''
        
        pairs_to_relevances[(query_id, doc_id)].append({
            'relevance': relevance,
            'note': note,
            'original_entry': qrels_list[i]
        })
    
    deduplicated_qrels = []
    
    for (query_id, doc_id), entries in pairs_to_relevances.items():
        relevance_scores = [entry['relevance'] for entry in entries]
        relevance_counter = Counter(relevance_scores)
        max_count = relevance_counter.most_common(1)[0][1]
        most_common_scores = [score for score, count in relevance_counter.items() if count == max_count]
        mode_relevance = max(most_common_scores)
        
        deduplicated_qrels.append({
            'query_id': query_id,
            'doc_id': doc_id,
            'relevance': mode_relevance,
            'count': len(entries)
        })

    query_to_docs = defaultdict(list)
    
    for entry in deduplicated_qrels:
        query_to_docs[entry['query_id']].append({
            'doc_id': entry['doc_id'],
            'relevance': entry['relevance'],
            'count': entry['count']
        })
    
    # Sort each query's docs by relevance (highest first)
    for query_id in query_to_docs:
        query_to_docs[query_id].sort(key=lambda x: x['relevance'], reverse=True)
    
    return dict(query_to_docs)

def main():
    """
    Main evaluation function
    """
    # Load the existing CodeSearchNet data with queries and ground truth
    with open(CONFIG['output']['input_data'], 'rb') as f:
        loaded_data = pickle.load(f)
    query_lookup = loaded_data['query_lookup']
    repo_mapping_sorted = loaded_data['repo_mapping_sorted']
    
    # Load entire corpus
    print("Loading entire CodeSearchNet corpus...")
    entire_corpus = ir_datasets.load("codesearchnet")
    
    print("Building doc_id to repo mapping for Python functions...")
    entire_corpus_doc_to_repo = {}
    entire_corpus_python_doc_ids = set()
    
    for doc in entire_corpus.docs_iter():
        if doc.language == 'python':
            entire_corpus_doc_to_repo[doc.doc_id] = doc.repo
            entire_corpus_python_doc_ids.add(doc.doc_id)
    
    print(f"Found {len(entire_corpus_python_doc_ids)} Python functions")
    
    # Group documents by repositories
    print("Grouping documents by repository...")
    docs_by_repo = defaultdict(list)
    
    for doc in entire_corpus.docs_iter():
        if doc.language == 'python':
            docs_by_repo[doc.repo].append(doc)
    
    print(f"Grouped documents into {len(docs_by_repo)} repositories")
    
    # Store all results
    evaluation_results = {
        'retrieval_method': 'vector_qwen3_8b',
        'repositories': {},
        'overall_stats': {}
    }
    
    all_ndcg_scores = []
    all_recall_scores = []
    total_queries = 0
    
    # Process each repository similar to CodeSearchNet_eval.py
    for repo_name in repo_mapping_sorted.keys():
        if repo_mapping_sorted[repo_name]['qrels_count'] < CONFIG['processing']['min_qrels_count']:
            break
            
        print("="*60)
        print(f"Processing repository: {repo_name}")
        print(f"Using Vector Database with Qwen3-8B embeddings")
        
        # Get documents for this repository
        repo_docs = docs_by_repo.get(repo_name, [])
        if not repo_docs:
            print(f"No documents found for repository {repo_name}")
            continue
            
        print(f"Found {len(repo_docs)} documents for repository {repo_name}")
        
        # Get query to docs mapping
        query_to_docs = deduplicate_qrels(repo_mapping_sorted[repo_name]['qrels'])
        
        # Create ChromaDB collection for this repository
        collection_name = f"codesearchnet_{repo_name.replace('/', '_')}"
        try:
            CHROMA_CLIENT.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist
            
        collection = CHROMA_CLIENT.create_collection(name=collection_name)
        
        # Build vector index for this repository
        if not build_vector_index_from_docs(repo_name, repo_docs, collection):
            print(f"Failed to build index for {repo_name}")
            continue
        
        repo_results = {
            'repo_name': repo_name,
            'total_docs': len(repo_docs),
            'queries': {},
            'repo_stats': {
                'total_queries': len(query_to_docs),
                'avg_ndcg_10': 0.0,
                'avg_recall_10': 0.0
            }
        }
        
        repo_ndcg_scores = []
        repo_recall_scores = []
        
        # Process each query in this repository
        for query_id in query_to_docs.keys():
            print(f"Processing query {query_id}")
            
            try:
                eval_result = evaluate_retrieval_vector(
                    query_id, query_lookup, query_to_docs, collection
                )
                
                if eval_result is None:
                    print(f"  Skipped query {query_id}: No relevant ground truth")
                    continue
                
                # Store query results
                query_result = {
                    'query_text': query_lookup[query_id][1],
                    'ndcg_10': eval_result['ndcg_10'],
                    'recall_10': eval_result['recall_10'],
                    'total_relevant': eval_result['total_relevant'],
                    'relevant_retrieved': eval_result['relevant_retrieved'],
                }
                
                repo_results['queries'][query_id] = query_result
                
                # Collect scores for averaging
                repo_ndcg_scores.append(eval_result['ndcg_10'])
                repo_recall_scores.append(eval_result['recall_10'])
                all_ndcg_scores.append(eval_result['ndcg_10'])
                all_recall_scores.append(eval_result['recall_10'])
                total_queries += 1
                
                print(f"  Query {query_id}: NDCG@10={eval_result['ndcg_10']:.4f}, Recall@10={eval_result['recall_10']:.4f}")
                
            except Exception as e:
                print(f"  Error processing query {query_id}: {str(e)}")
                continue
        
        # Calculate repository averages
        if repo_ndcg_scores:
            repo_results['repo_stats']['avg_ndcg_10'] = sum(repo_ndcg_scores) / len(repo_ndcg_scores)
            repo_results['repo_stats']['avg_recall_10'] = sum(repo_recall_scores) / len(repo_recall_scores)
        
        evaluation_results['repositories'][repo_name] = repo_results
        
        # Update overall statistics
        evaluation_results['overall_stats'] = {
            'total_repositories': len(evaluation_results['repositories']),
            'total_queries': total_queries,
            'overall_avg_ndcg_10': sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0,
            'overall_avg_recall_10': sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
        }
        
        # Save results after each repository
        output_filename = CONFIG['output']['vector_results']
        with open(output_filename, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Repository {repo_name} completed:")
        print(f"  Avg NDCG@10: {repo_results['repo_stats']['avg_ndcg_10']:.4f}")
        print(f"  Avg Recall@10: {repo_results['repo_stats']['avg_recall_10']:.4f}")
        print(f"Results updated in {output_filename}")
        
        # Clean up vector database for this repository to free memory
        try:
            CHROMA_CLIENT.delete_collection(name=collection_name)
            print(f"  ✓ Cleaned up vector database for {repo_name}")
        except Exception as e:
            print(f"  Warning: Could not clean up collection {collection_name}: {e}")
        
        # Also clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  ✓ Cleared GPU cache")
    
    # Final results
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print(f"Total repositories processed: {evaluation_results['overall_stats']['total_repositories']}")
    print(f"Total queries processed: {evaluation_results['overall_stats']['total_queries']}")
    print(f"Overall average NDCG@10: {evaluation_results['overall_stats']['overall_avg_ndcg_10']:.4f}")
    print(f"Overall average Recall@10: {evaluation_results['overall_stats']['overall_avg_recall_10']:.4f}")
    
    # Final save
    output_filename = CONFIG['output']['vector_results']
    with open(output_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Final results saved to: {output_filename}")

if __name__ == "__main__":
    main() 