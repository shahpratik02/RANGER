__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pickle
import math
import time
from collections import defaultdict, Counter
from src.core.MCTS_cross_encoder_batch_new import RLEnhancedGraphRAG, main
from src.utils.config import CONFIG
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer
import json
import os
# Add semantic retriever import
from src.core.semantic_retriever import main as semantic_main
# Add ChromaDB baseline imports
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ChromaDB baseline configuration
EMBEDDING_MODEL = CONFIG['models']['qwen_embedding']
CHROMA_CLIENT = chromadb.Client()

# Global model variables
_model = None
_tokenizer = None


def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_model():
    """Load and return the Qwen embedding model"""
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


def embed_texts_qwen(texts, batch_size=32):
    """Embed texts using Qwen model with batching"""
    model, tokenizer = get_model()
    embeddings = []
    if len(texts) > 3000:
        batch_size = min(batch_size, 16)
        print(f"  Large dataset detected ({len(texts)} texts), reducing batch size to {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
        # Normalize embeddings
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings.extend(batch_embeddings.cpu().numpy().tolist())
    
    return embeddings


# def get_node_path_to_root(node):
#     """
#     Trace the path from a TreeNode back to the root node
    
#     Args:
#         node: TreeNode object to trace from
        
#     Returns:
#         List of node dictionaries from root to current node (inverted path)
#     """
#     path = []
#     current = node
    
#     while current is not None:
#         node_id = current.graph_node_id
#         node_name = current.graph_node_data.get('name', 'Unknown')
#         node_type = current.graph_node_data.get('node_type', 'Unknown')
#         signature = current.graph_node_data.get('signature', '')
#         module_name = current.graph_node_data.get('module_name', '')
        
#         # Get simulation reward (reward from direct evaluations of this node)
#         sim_reward = current.simulation_reward / current.simulation_visits if current.simulation_visits > 0 else 0.0
        
#         path.append({
#             'node_id': node_id,
#             'name': node_name,
#             'type': node_type,
#             'signature': signature,
#             'module_name': module_name,
#             'avg_reward': current.get_average_reward() if current.visit_count > 0 else 0.0,
#             'sim_reward': sim_reward,
#             'visit_count': current.visit_count,
#             'simulation_visits': current.simulation_visits
#         })
#         current = current.parent
    
#     # Reverse the path to go from root to node
#     path.reverse()
#     return path


def store_match_paths(match_paths, filename=None):
    """
    Store match path information to a JSON file
    
    Args:
        match_paths: List of match path dictionaries
        filename: Output filename (defaults to config value)
    """
    if filename is None:
        filename = CONFIG['output']['match_paths']
    
    try:
        # Try to load existing data
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # Append new match paths
        existing_data.extend(match_paths)
        
        # Save updated data
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Stored {len(match_paths)} match paths in {filename}")
        
    except Exception as e:
        print(f"Error storing match paths: {e}")


def path_to_module_name(path):
    """
    Convert file path to module name.
    Example: 'src/package/module.py' -> 'src.package.module'
    """
    # Remove .py extension
    if path.endswith(".py"):
        path = path[:-3]

    # Convert path separators to dots
    module_name = path.replace("/", ".").replace("\\", ".")

    # Handle __init__.py files - remove the __init__ part
    if module_name.endswith(".__init__"):
        module_name = module_name[:-9]

    return module_name

def extract_purpose(text):
    lines = text.split('\n')
    for line in lines:
        if 'Purpose' in line:
            # Find the position of 'Purpose' and extract everything after it
            purpose_idx = line.find('Purpose')
            if purpose_idx != -1:
                # Look for ':' after 'Purpose'
                colon_idx = line.find(':', purpose_idx)
                if colon_idx != -1:
                    return line[colon_idx + 1:].strip()
                else:
                    # If no colon, take everything after 'Purpose'
                    return line[purpose_idx + 7:].strip()  # 7 is len('Purpose')
    return ""


def build_chromadb_index(repo_name, neo4j_config):
    """
    Build ChromaDB index from Neo4j function/method nodes using code content
    """
    print(f"Building ChromaDB index for repository: {repo_name}")
    
    # Connect to Neo4j
    try:
        graph = Neo4jGraph(
            url=neo4j_config["url"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
        )
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return None
    
    # Query to fetch all function/method nodes with their code
    query = """
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS]-(:Module)-[:CONTAINS]-(c)
    WHERE c:Function OR c:GlobalVariable
    RETURN id(c) AS id, c.name AS name, c.signature AS signature, c.code AS code, labels(c) AS type, c.module_name AS module, null AS class_name
    
    UNION
    
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS]-(:Module)-[:CONTAINS]-(c)-[:HAS_METHOD]-(m:Method)
    RETURN id(m) AS id, m.name AS name, m.signature AS signature, m.code AS code, labels(m) AS type, m.module_name AS module, c.name AS class_name
    """
    
    try:
        nodes = graph.query(query, params={"repo_name": repo_name})
        print(f"  Found {len(nodes)} nodes to index.")
    except Exception as e:
        print(f"  Error querying graph: {e}")
        graph.close()
        return None
    
    # Filter nodes that have code
    nodes_with_code = [node for node in nodes if node.get("code") and node.get("code").strip()]
    print(f"  Found {len(nodes_with_code)} nodes with code content.")
    
    if not nodes_with_code:
        print("  No nodes with code found.")
        graph.close()
        return None
    
    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50
    )
    
    # Process nodes and create chunks
    documents = []
    metadatas = []
    ids = []
    
    for node in nodes_with_code:
        code = node.get("code", "")
        if not code.strip():
            continue
            
        node_type_list = node.get("type", [])
        
        # Split code into chunks
        code_chunks = text_splitter.split_text(code)
        
        for i, chunk in enumerate(code_chunks):
            metadata = {
                "name": node.get("name") or "",
                "type": ",".join(node_type_list),
                "module_name": node.get("module") or "",
                "class_name": node.get("class_name") or "",
                "repo_name": repo_name,
                "original_node_id": str(node.get("id") or ""),
                "chunk_index": i,
                "total_chunks": len(code_chunks)
            }
            
            if "GlobalVariable" not in node_type_list:
                metadata["signature"] = node.get("signature") or ""
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{node.get('id')}_{i}")
    
    if not documents:
        print("  No documents to index.")
        graph.close()
        return None
    
    # Create ChromaDB collection
    safe_repo_name = repo_name.replace('/', '_').replace(' ', '_').replace('-', '_')
    collection_name = f"code_baseline_{safe_repo_name}"
    
    try:
        # Delete existing collection if it exists
        try:
            CHROMA_CLIENT.delete_collection(name=collection_name)
        except:
            pass
        
        collection = CHROMA_CLIENT.create_collection(name=collection_name)
        
        # Embed documents
        print(f"  Embedding {len(documents)} code chunks...")
        embeddings = embed_texts_qwen(documents)
        
        # Add to ChromaDB
        # Add in batches to avoid exceeding backend max batch size
        max_batch_size = 2000
        print(f"  Adding to ChromaDB in batches of {max_batch_size}...")
        for start_idx in range(0, len(documents), max_batch_size):
            end_idx = start_idx + max_batch_size
            collection.add(
                documents=documents[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],
                ids=ids[start_idx:end_idx],
                embeddings=embeddings[start_idx:end_idx]
            )
        
        print(f"  ✓ ChromaDB index built successfully with {len(documents)} chunks.")
        graph.close()
        return collection
        
    except Exception as e:
        print(f"  Error building ChromaDB index: {e}")
        graph.close()
        return None


def evaluate_retrieval_mcts(needle_data, rl_graphrag, repo_name):
    """Evaluation function for MCTS retriever adapted for RepoQA needles"""
    
    description = needle_data['description']
    target_function_name = needle_data['name']
    target_file_path = needle_data['path']
    purpose=extract_purpose(description)
    
    # Measure MCTS search time
    mcts_start_time = time.time()
    res = rl_graphrag.search(purpose, repo_name=repo_name)
    mcts_end_time = time.time()
    mcts_execution_time = mcts_end_time - mcts_start_time
    
    # Store match paths for this query
    match_paths = []
    
    print(f"\nSearching for function '{target_function_name}' in '{target_file_path}'")
    print(f"Query: {description}")
    print(f"MCTS execution time: {mcts_execution_time:.4f} seconds")
    print(f"Found {len(res)} high-reward nodes:")
    for i, node_info in enumerate(res, 1):
        print(
            f"{i}. {node_info['node_data'].get('name', 'Unknown')} "
            f"(Reward: {node_info['avg_reward']:.2f}, "
            f"Visits: {node_info['visit_count']})"
        )
    
    # Convert target file path to module name for comparison
    target_module_name = path_to_module_name(target_file_path)
    
    # Extract names from retrieved results and check for matches
    retrieved_relevances = []
    found_match = False
    
    for i in range(min(10, len(res))):  # Only consider top 10
        relevance = 0
        node_data = res[i]['node_data']
        tree_node = res[i].get('tree_node')  # Get the TreeNode object
        
        # Extract the name based on node type
        retrieved_function_name = node_data.get('name', '')
        retrieved_module_name = node_data.get('module_name', '')
        
        # If the retrieved node is a Class, expand its methods and evaluate against the target
        try:
            node_type = node_data.get('node_type')
            labels = node_data.get('labels') or []
            if tree_node and not node_type:
                node_type = getattr(tree_node, 'graph_node_data', {}).get('node_type')
            if tree_node and not labels:
                labels = getattr(tree_node, 'graph_node_data', {}).get('labels') or []
        except Exception:
            node_type = None
            labels = []
        
        is_class_node = (node_type == 'Class') or (isinstance(labels, list) and 'Class' in labels)
        if is_class_node and tree_node and hasattr(tree_node, 'graph_node_id'):
            methods = []
            graph_obj = getattr(rl_graphrag, 'graph', None)
            if graph_obj:
                try:
                    methods = graph_obj.query(
                            """
                            MATCH (c:Class {name: $name, signature: $signature, module_name: $module_name})
                            -[:HAS_METHOD]-(m:Method)
                            RETURN m.name AS name, m.module_name AS module_name, c.name AS class_name
                            """,
                            params={
                                "name": tree_node.graph_node_data.get('name'),
                                "signature": tree_node.graph_node_data.get('signature', ''),
                                "module_name": tree_node.graph_node_data.get('module_name', '')
                            }
                        )
                except Exception as e:
                    print(f"  Warning: failed to expand class methods: {e}")
            
            for m in methods:
                method_name = m.get('name', '')
                method_module = m.get('module_name', '')
                method_class = m.get('class_name', '')
                class_match = False
                
                if '.' in target_function_name:
                    target_class, target_method = target_function_name.rsplit('.', 1)
                    if (
                        method_name == target_method
                        and method_class == target_class
                        and (not method_module or method_module == target_module_name)
                    ):
                        class_match = True
                else:
                    if method_name == target_function_name and (not method_module or method_module == target_module_name):
                        class_match = True
                
                if class_match:
                    relevance = 1
                    found_match = True
                    if tree_node:
                        match_info = {
                            'description': description,
                            'target_function': target_function_name,
                            'target_file': target_file_path,
                            'target_module': target_module_name,
                            'repo_name': repo_name,
                            'retrieved_function': f"{method_class}.{method_name}",
                            'retrieved_module': method_module,
                            'relevance': relevance,
                            'rank': i + 1,
                        }
                        match_paths.append(match_info)
                        print(f"CLASS CONTAINS MATCH: {target_function_name} at rank {i+1}")
                    # Do not break outer loop; we still append relevance for this rank below
                    break
        
        # Check for exact function name match
        if retrieved_function_name == target_function_name:
            # Also check if it's in the correct module (if module info is available)
            if not retrieved_module_name or retrieved_module_name == target_module_name:
                relevance = 1  # Binary relevance for RepoQA
                found_match = True
                
                # Store the path for this match
                if tree_node:
                    # path = get_node_path_to_root(tree_node)
                    match_info = {
                        'description': description,
                        'target_function': target_function_name,
                        'target_file': target_file_path,
                        'target_module': target_module_name,
                        'repo_name': repo_name,
                        'retrieved_function': retrieved_function_name,
                        'retrieved_module': retrieved_module_name,
                        'relevance': relevance,
                        'rank': i + 1,
                        # 'path': path
                    }
                    match_paths.append(match_info)
                    print(f"EXACT MATCH FOUND: {target_function_name} at rank {i+1}")
        
        # Also check for method matches (class.method pattern)
        elif '.' in target_function_name:
            class_name, method_name = target_function_name.rsplit('.', 1)
            if retrieved_function_name == method_name:
                # Check if this method belongs to the right class
                retrieved_class = node_data.get('class', '')
                if retrieved_class == class_name:
                    relevance = 1
                    found_match = True
                    
                    # Store the path for this method match
                    if tree_node:
                        # path = get_node_path_to_root(tree_node)
                        match_info = {
                            'description': description,
                            'target_function': target_function_name,
                            'target_file': target_file_path,
                            'target_module': target_module_name,
                            'repo_name': repo_name,
                            'retrieved_function': f"{retrieved_class}.{retrieved_function_name}",
                            'retrieved_module': retrieved_module_name,
                            'relevance': relevance,
                            'rank': i + 1,
                        }
                        match_paths.append(match_info)
                        print(f"METHOD MATCH FOUND: {target_function_name} at rank {i+1}")
        
        retrieved_relevances.append(relevance)
        
        # Break after finding the first match to ensure only one relevant result
        if relevance == 1:
            break
    
    # Store the match paths for this query
    if match_paths:
        safe_repo_name = repo_name.replace('/', '_').replace(' ', '_')
        # store_match_paths(match_paths, filename=f"traces/repoqa_match_paths_{safe_repo_name}.json")
    
    # Pad with zeros if we have less than 10 results
    while len(retrieved_relevances) < 10:
        retrieved_relevances.append(0)
    
    # Calculate metrics
    def calculate_dcg(relevances):
        dcg = 0.0
        for i, rel in enumerate(relevances):
            if rel > 0:
                dcg += rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
        return dcg
    
    # DCG for retrieved results
    dcg = calculate_dcg(retrieved_relevances)
    
    # IDCG - ideal DCG (1 relevant item at position 1)
    ideal_relevances = [1] + [0] * 9  # Only one relevant item per needle
    idcg = calculate_dcg(ideal_relevances)
    
    # NDCG@10
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0
    
    # Calculate Recall@10 (binary: either we found it or we didn't)
    recall_10 = 1.0 if found_match else 0.0
    
    return {
        'ndcg_10': ndcg_10,
        'recall_10': recall_10,
        'total_relevant': 1,  # Always 1 for needle tasks
        'relevant_retrieved': 1 if found_match else 0,
        'target_function': target_function_name,
        'target_file': target_file_path,
        'found_match': found_match,
        'mcts_execution_time': mcts_execution_time
    }

def evaluate_retrieval_semantic(needle_data, repo_name):
    """Evaluation function for semantic retriever adapted for RepoQA needles"""
    
    description = needle_data['description']
    target_function_name = needle_data['name']
    target_file_path = needle_data['path']
    
    # Neo4j config for semantic retriever
    neo4j_config = CONFIG['neo4j']
    
    # Get results from semantic retriever
    semantic_results = semantic_main(question=description, repo_name=repo_name, neo4j_config=neo4j_config)
    
    print(f"\nSearching for function '{target_function_name}' in '{target_file_path}'")
    print(f"Query: {description}")
    print(f"Found {len(semantic_results)} semantic results")
    
    # Convert target file path to module name for comparison
    target_module_name = path_to_module_name(target_file_path)
    
    # Extract names from retrieved results and check for matches
    retrieved_relevances = []
    found_match = False
    
    for i in range(min(10, len(semantic_results))):  # Only consider top 10
        relevance = 0
        result = semantic_results[i]
        
        # Extract the name - use class_name field directly
        func_name = result.get('name', '')
        class_name = result.get('class_name')
        module_name = result.get('module_name', '')
        
        print(f"  {i+1}. {func_name} (class: {class_name}, module: {module_name})")
        
        # Check for exact function name match
        if func_name == target_function_name:
            # Also check if it's in the correct module (if module info is available)
            if not module_name or module_name == target_module_name:
                relevance = 1
                found_match = True
                print(f"EXACT MATCH FOUND: {target_function_name} at rank {i+1}")
        
        # Also check for method matches (class.method pattern)
        elif '.' in target_function_name:
            target_class_name, target_method_name = target_function_name.rsplit('.', 1)
            if func_name == target_method_name and class_name == target_class_name:
                relevance = 1
                found_match = True
                print(f"METHOD MATCH FOUND: {target_function_name} at rank {i+1}")
        
        retrieved_relevances.append(relevance)

        if relevance == 1:
            break
    
    # Pad with zeros if we have less than 10 results
    while len(retrieved_relevances) < 10:
        retrieved_relevances.append(0)
    
    # Calculate metrics (same as MCTS version)
    def calculate_dcg(relevances):
        dcg = 0.0
        for i, rel in enumerate(relevances):
            if rel > 0:
                dcg += rel / math.log2(i + 2)
        return dcg
    
    dcg = calculate_dcg(retrieved_relevances)
    ideal_relevances = [1] + [0] * 9  # Only one relevant item per needle
    idcg = calculate_dcg(ideal_relevances)
    
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0
    recall_10 = 1.0 if found_match else 0.0
    
    return {
        'ndcg_10': ndcg_10,
        'recall_10': recall_10,
        'total_relevant': 1,
        'relevant_retrieved': 1 if found_match else 0,
        'target_function': target_function_name,
        'target_file': target_file_path,
        'found_match': found_match
    }


def evaluate_retrieval_code_semantic(needle_data, repo_name, collection):
    """Evaluation function for code-based semantic retriever using ChromaDB baseline"""
    
    description = needle_data['description']
    target_function_name = needle_data['name']
    target_file_path = needle_data['path']
    purpose = extract_purpose(description)
    
    # Use the pre-built collection
    if not collection:
        print(f"No ChromaDB collection available for {repo_name}")
        return {
            'ndcg_10': 0.0,
            'recall_10': 0.0,
            'total_relevant': 1,
            'relevant_retrieved': 0,
            'target_function': target_function_name,
            'target_file': target_file_path,
            'found_match': False
        }
    
    # Query ChromaDB
    query_text = purpose if purpose else description
    query_embeddings = embed_texts_qwen([query_text])
    
    try:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return {
            'ndcg_10': 0.0,
            'recall_10': 0.0,
            'total_relevant': 1,
            'relevant_retrieved': 0,
            'target_function': target_function_name,
            'target_file': target_file_path,
            'found_match': False
        }
    
    print(f"\nSearching for function '{target_function_name}' in '{target_file_path}'")
    print(f"Query: {description}")
    print(f"Found {len(results['metadatas'][0])} code-based semantic results:")
    
    # Convert target file path to module name for comparison
    target_module_name = path_to_module_name(target_file_path)
    
    # Extract names from retrieved results and check for matches
    retrieved_relevances = []
    found_match = False
    
    for i in range(min(10, len(results['metadatas'][0]))):
        relevance = 0
        metadata = results['metadatas'][0][i]
        
        retrieved_function_name = metadata.get('name', '')
        retrieved_module_name = metadata.get('module_name', '')
        retrieved_class_name = metadata.get('class_name', '')
        
        print(f"  {i+1}. {retrieved_function_name} (module: {retrieved_module_name}, class: {retrieved_class_name})")
        
        # Check for exact function name match
        if retrieved_function_name == target_function_name:
            # Also check if it's in the correct module (if module info is available)
            if not retrieved_module_name or retrieved_module_name == target_module_name:
                relevance = 1
                found_match = True
                print(f"EXACT MATCH FOUND: {target_function_name} at rank {i+1}")
        
        # Also check for method matches (class.method pattern)
        elif '.' in target_function_name:
            target_class_name, target_method_name = target_function_name.rsplit('.', 1)
            if retrieved_function_name == target_method_name and retrieved_class_name == target_class_name:
                relevance = 1
                found_match = True
                print(f"METHOD MATCH FOUND: {target_function_name} at rank {i+1}")
        
        retrieved_relevances.append(relevance)
        
        # Break after finding the first match to ensure only one relevant result
        if relevance == 1:
            break
    
    # Pad with zeros if we have less than 10 results
    while len(retrieved_relevances) < 10:
        retrieved_relevances.append(0)
    
    # Calculate metrics (same as other methods)
    def calculate_dcg(relevances):
        dcg = 0.0
        for i, rel in enumerate(relevances):
            if rel > 0:
                dcg += rel / math.log2(i + 2)
        return dcg
    
    dcg = calculate_dcg(retrieved_relevances)
    ideal_relevances = [1] + [0] * 9  # Only one relevant item per needle
    idcg = calculate_dcg(ideal_relevances)
    
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0
    recall_10 = 1.0 if found_match else 0.0
    
    return {
        'ndcg_10': ndcg_10,
        'recall_10': recall_10,
        'total_relevant': 1,
        'relevant_retrieved': 1 if found_match else 0,
        'target_function': target_function_name,
        'target_file': target_file_path,
        'found_match': found_match
    }


def main(use_code_semantic=False, use_text_semantic=False):
    # Load RepoQA data
    print("Loading RepoQA dataset...")
    data = load_jsonl(CONFIG['paths']['input_data'])
    python_repos = data[0]['python']
    print(f"Found {len(python_repos)} Python repositories")
    
    # Neo4j configuration
    graph_config = CONFIG['neo4j']
    
    graph = Neo4jGraph(**graph_config)
    embedding_model = SentenceTransformer(CONFIG['models']['embedding'])
    
    # Initialize retriever (only needed for MCTS mode)
    rl_graphrag = None
    if not use_code_semantic and not use_text_semantic:
        rl_graphrag = RLEnhancedGraphRAG(
            graph=graph,
            embedding_model=embedding_model,
            max_iterations=CONFIG['mcts']['max_iterations'],
            reward_threshold=CONFIG['mcts']['reward_threshold'],
            alpha=CONFIG['mcts']['alpha'],
            cross_encoder_model=CONFIG['models']['cross_encoder'],
            top_k_children=CONFIG['mcts']['top_k_children'],
            top_k_references=CONFIG['mcts']['top_k_references'],
            reduce_top_k_flag=CONFIG['mcts']['reduce_top_k_flag'],
            min_top_k_children=CONFIG['mcts']['min_top_k_children'],
            exploration_param=CONFIG['mcts']['exploration_param']
        )
    
    # Store all results
    retrieval_method = 'code_semantic' if use_code_semantic else ('text_semantic' if use_text_semantic else 'mcts')
    evaluation_results = {
        'retrieval_method': retrieval_method,
        'repositories': {},
        'overall_stats': {}
    }
    
    all_ndcg_scores = []
    all_recall_scores = []
    all_mcts_times = []
    total_needles = 0
    
    # Base directory for RepoQA clones  
    base_clone_dir = os.path.join(CONFIG['paths']['data_dir'], "RepoQA/")
    
    # Process repositories
    for i, repo_data in enumerate(python_repos):  # Process first 10 repositories
        repo_path_raw = repo_data['repo']
        commit_sha = repo_data['commit_sha']
        content_files = repo_data['content']
        dependency_files = repo_data['dependency']
        needles_list = repo_data.get('needles', [])  # Use needles instead of qa
        
        if not needles_list:
            print(f"No needles found for repository {repo_path_raw} - skipping")
            continue
        
        # Extract repo name
        if repo_path_raw.startswith('https://'):
            repo_url = repo_path_raw
        else:
            repo_url = f"https://github.com/{repo_path_raw}.git"
        
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        if repo_url.count('/') >= 4:
            owner = repo_url.split('/')[-2]
            repo_name = f"{owner}_{repo_name}"
        
        repo_path = os.path.join(base_clone_dir, repo_name)
        
        print("="*60)
        print(f"Processing repository {i+1}: {repo_name}")
        print(f"Number of needles: {len(needles_list)}")
        method_name = 'Code-based Semantic Retriever' if use_code_semantic else ('Text-based Semantic Retriever' if use_text_semantic else 'MCTS')
        print(f"Using {method_name}")
        print("="*60)
        
        # Check if repository exists in graph
        cypher_query = f"""
        MATCH (r:Repo {{name:'{repo_path}'}})
        RETURN r
        """
        result = graph.query(cypher_query)
        if not result:
            print(f"Repository {repo_name} not found in graph - skipping")
            continue
        
        # Get total number of nodes in the graph for this repository
        cypher_query = f"""
        MATCH (r:Repo {{name:'{repo_path}'}})-[:CONTAINS]-(mod:Module)-[:CONTAINS]-(c)
        OPTIONAL MATCH (c)-[:HAS_METHOD]-(m)
        RETURN count(c)+count(m)+1 as total_nodes
        """
        result = graph.query(cypher_query)
        total_graph_nodes = result[0]['total_nodes'] if result else 0
        print(f"Total graph nodes for {repo_name}: {total_graph_nodes}")
        if total_graph_nodes <= 1:
            print(f"Repository {repo_name} has insufficient nodes - skipping")
            continue
        
        # Adjust MCTS parameters based on repository size (only for MCTS)
        if not use_code_semantic and not use_text_semantic and rl_graphrag:
            module_count_query = f"""
            MATCH (r:Repo {{name:'{repo_path}'}})-[:CONTAINS]->(m:Module)
            RETURN count(m) as module_count
            """
            module_result = graph.query(module_count_query)
            module_count = module_result[0]['module_count'] if module_result else 100
            
            if module_count < 20:
                rl_graphrag.original_top_k_children = max(1, module_count // 2)
            else:
                rl_graphrag.original_top_k_children = min(round(module_count / 10) * 5, 200) + 15
            
            rl_graphrag.exploration_param = 1 / (1 * math.sqrt(math.log(4 * rl_graphrag.original_top_k_children)))
            print(f"MCTS exploration_param: {rl_graphrag.exploration_param}")
        
        repo_results = {
            'repo_name': repo_name,
            'total_graph_nodes': total_graph_nodes,
            'needles': {},
            'repo_stats': {
                'total_needles': len(needles_list),
                'avg_ndcg_10': 0.0,
                'avg_recall_10': 0.0,
                'avg_mcts_execution_time': 0.0
            }
        }
        
        repo_ndcg_scores = []
        repo_recall_scores = []
        repo_mcts_times = []
        
        # Build ChromaDB index once per repository (only for code_semantic mode)
        chromadb_collection = None
        if use_code_semantic:
            print(f"Building ChromaDB index for repository: {repo_name}")
            chromadb_collection = build_chromadb_index(repo_path, graph_config)
            if not chromadb_collection:
                print(f"Failed to build ChromaDB index for {repo_name} - skipping")
                continue
        
        # Process each needle in this repository
        for needle_idx, needle_data in enumerate(needles_list):
            print(f"Processing needle {needle_idx + 1}/{len(needles_list)}")
            
            try:
                # Choose evaluation function based on retriever type
                if use_code_semantic:
                    eval_result = evaluate_retrieval_code_semantic(needle_data, repo_path, chromadb_collection)
                elif use_text_semantic:
                    eval_result = evaluate_retrieval_semantic(needle_data, repo_path)
                else:
                    eval_result = evaluate_retrieval_mcts(needle_data, rl_graphrag, repo_path)
                
                # Count nodes in MCTS tree (only for MCTS)
                mcts_node_count = 0
                if not use_code_semantic and not use_text_semantic and hasattr(rl_graphrag, 'root_tree_node') and rl_graphrag.root_tree_node:
                    queue = [rl_graphrag.root_tree_node]
                    while queue:
                        node = queue.pop(0)
                        mcts_node_count += 1
                        if hasattr(node, 'children'):
                            queue.extend(node.children)
                
                # Store needle results
                needle_result = {
                    'description': needle_data['description'],
                    'target_function': needle_data['name'],
                    'target_file': needle_data['path'],
                    'ndcg_10': eval_result['ndcg_10'],
                    'recall_10': eval_result['recall_10'],
                    'total_relevant': eval_result['total_relevant'],
                    'relevant_retrieved': eval_result['relevant_retrieved'],
                    'found_match': eval_result['found_match']
                }
                
                if not use_code_semantic and not use_text_semantic:
                    needle_result['mcts_tree_nodes'] = mcts_node_count
                    needle_result['mcts_execution_time'] = eval_result.get('mcts_execution_time', 0.0)
                
                repo_results['needles'][needle_idx] = needle_result
                
                # Collect scores for averaging
                repo_ndcg_scores.append(eval_result['ndcg_10'])
                repo_recall_scores.append(eval_result['recall_10'])
                if not use_code_semantic and not use_text_semantic:
                    repo_mcts_times.append(eval_result.get('mcts_execution_time', 0.0))
                all_ndcg_scores.append(eval_result['ndcg_10'])
                all_recall_scores.append(eval_result['recall_10'])
                if not use_code_semantic and not use_text_semantic:
                    all_mcts_times.append(eval_result.get('mcts_execution_time', 0.0))
                total_needles += 1
                
                # Print progress
                node_info = f", MCTS nodes={mcts_node_count}" if not use_code_semantic and not use_text_semantic else ""
                time_info = f", Time={eval_result.get('mcts_execution_time', 0.0):.4f}s" if not use_code_semantic and not use_text_semantic else ""
                print(f"  Needle {needle_idx + 1}: NDCG@10={eval_result['ndcg_10']:.4f}, Recall@10={eval_result['recall_10']:.4f}{node_info}{time_info}")
                
            except Exception as e:
                print(f"  Error processing needle {needle_idx + 1}: {str(e)}")
                continue
        
        # Calculate repository averages
        if repo_ndcg_scores:
            repo_results['repo_stats']['avg_ndcg_10'] = sum(repo_ndcg_scores) / len(repo_ndcg_scores)
            repo_results['repo_stats']['avg_recall_10'] = sum(repo_recall_scores) / len(repo_recall_scores)
            if repo_mcts_times:
                repo_results['repo_stats']['avg_mcts_execution_time'] = sum(repo_mcts_times) / len(repo_mcts_times)
        
        evaluation_results['repositories'][repo_name] = repo_results
        
        # Update overall statistics with current progress
        evaluation_results['overall_stats'] = {
            'total_repositories': len(evaluation_results['repositories']),
            'total_needles': total_needles,
            'overall_avg_ndcg_10': sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0,
            'overall_avg_recall_10': sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0,
            'overall_avg_mcts_execution_time': sum(all_mcts_times) / len(all_mcts_times) if all_mcts_times else 0.0
        }
        
        # Save results to JSON file after each repository
        if use_code_semantic:
            output_filename = CONFIG['output']['code_semantic_results']
        elif use_text_semantic:
            output_filename = CONFIG['output']['text_semantic_results']
        else:
            output_filename = CONFIG['output']['mcts_results']
        with open(output_filename, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Repository {repo_name} completed:")
        print(f"  Avg NDCG@10: {repo_results['repo_stats']['avg_ndcg_10']:.4f}")
        print(f"  Avg Recall@10: {repo_results['repo_stats']['avg_recall_10']:.4f}")
        if not use_code_semantic and not use_text_semantic:
            print(f"  Avg MCTS execution time: {repo_results['repo_stats']['avg_mcts_execution_time']:.4f} seconds")
        print(f"  Total graph nodes: {total_graph_nodes}")
        print(f"Results updated in {output_filename}")
        
        # Clear ChromaDB after each repository to avoid accumulation
        if use_code_semantic and chromadb_collection:
            try:
                safe_repo_name = repo_path.replace('/', '_').replace(' ', '_').replace('-', '_')
                collection_name = f"code_baseline_{safe_repo_name}"
                CHROMA_CLIENT.delete_collection(name=collection_name)
                print(f"✓ Cleared ChromaDB collection for {repo_name}")
            except Exception as e:
                print(f"Warning: Could not clear ChromaDB collection: {e}")
        
        
        # Clear GPU memory after each repository
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ Cleared GPU memory cache after {repo_name}")
    
    # Final save to JSON file
    if use_code_semantic:
        output_filename = CONFIG['output']['code_semantic_results']
    elif use_text_semantic:
        output_filename = CONFIG['output']['text_semantic_results']
    else:
        output_filename = CONFIG['output']['mcts_results']
    with open(output_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print(f"Total repositories processed: {evaluation_results['overall_stats']['total_repositories']}")
    print(f"Total needles processed: {evaluation_results['overall_stats']['total_needles']}")
    print(f"Overall average NDCG@10: {evaluation_results['overall_stats']['overall_avg_ndcg_10']:.4f}")
    print(f"Overall average Recall@10: {evaluation_results['overall_stats']['overall_avg_recall_10']:.4f}")
    if not use_code_semantic and not use_text_semantic:
        print(f"Overall average MCTS execution time: {evaluation_results['overall_stats']['overall_avg_mcts_execution_time']:.4f} seconds")
    print(f"Final results saved to: {output_filename}")
    
    # Close graph connection
    graph.close()
    print("Graph connection closed.")


if __name__ == "__main__":
    # Add command line argument or simple toggle
    import sys
    use_code_semantic = len(sys.argv) > 1 and sys.argv[1] == '--code_semantic'
    use_text_semantic = len(sys.argv) > 1 and sys.argv[1] == '--text_semantic'
    
    if use_code_semantic:
        print("Using Code-based Semantic Retriever")
    elif use_text_semantic:
        print("Using Text-based Semantic Retriever")
    else:
        print("Using MCTS Retriever")
    
    main(use_code_semantic=use_code_semantic, use_text_semantic=use_text_semantic)