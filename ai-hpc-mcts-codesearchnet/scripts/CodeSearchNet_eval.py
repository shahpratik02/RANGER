import pickle
import math  
from collections import defaultdict, Counter
from src.core.MCTS_cross_encoder_batch_new import RLEnhancedGraphRAG, main
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer
import json
import os
from src.core.semantic_retriever import main as semantic_main
from src.utils.simple_config import CONFIG


def evaluate_retrieval_mcts(query_id, query_lookup, doc_lookup, query_to_docs, rl_graphrag, repo_name):
    """Original evaluation function for MCTS retriever"""
    query = query_lookup[query_id][1]
    res = rl_graphrag.search(query, repo_name=repo_name)
    
    # Store the complete iteration selections for this query
    if hasattr(rl_graphrag, 'selected_nodes_per_iteration') and rl_graphrag.selected_nodes_per_iteration:
        safe_repo_name = repo_name.replace('/', '_').replace(' ', '_')
        iteration_data = {
            'query_id': query_id,
            'query_text': query,
            'repo_name': repo_name,
            'selected_nodes_per_iteration': rl_graphrag.selected_nodes_per_iteration
        }
        iterations_file = os.path.join(CONFIG['paths']['traces_dir'], f"iterations_{safe_repo_name}_query_{query_id}.json")
        with open(iterations_file, 'w') as f:
            json.dump(iteration_data, f, indent=2)
        print(f"Stored {len(rl_graphrag.selected_nodes_per_iteration)} iteration selections for query {query_id}")
    
    # Store match paths for this query
    # match_paths = []
    
    print(f"\nFound {len(res)} high-reward nodes:")
    for i, node_info in enumerate(res, 1):
        print(
            f"{i}. {node_info['node_data'].get('name', 'Unknown')} "
            f"(Reward: {node_info['avg_reward']:.2f}, "
            f"Visits: {node_info['visit_count']})"
        )
    # Get ground truth documents for this query
    ground_truth_docs = query_to_docs.get(query_id, [])
    
    # Create a mapping from function/method names to relevance scores
    name_to_relevance = {}
    
    for doc_info in ground_truth_docs:
        doc_id = doc_info['doc_id']
        relevance = int(doc_info['relevance'])
        
        # Get the document and extract the function name
        doc = doc_lookup[doc_id]
        func_name = doc[3]  # Function name is at index 3
        
        name_to_relevance[func_name] = relevance
    
    # Check if there are any relevant ground truth entries
    relevant_gt_count = sum(1 for rel in name_to_relevance.values() if rel > 0)
    if relevant_gt_count == 0:
        print(f"Skipping query {query_id}: No relevant ground truth entries")
        return None
    
    # Extract names from retrieved results and create relevance vector
    retrieved_relevances = []
    matched_gt_items = set()  # Track which ground truth items have been matched
    
    for i in range(min(10, len(res))):  # Only consider top 10
        relevance = 0
        matched_gt_name = None
        node_data = res[i]['node_data']
        tree_node = res[i].get('tree_node')  # Get the TreeNode object
        
        # Extract the name based on node type
        if 'class' in node_data and node_data['class']:
            # This is a method - combine class and method name
            retrieved_name = node_data['class']+"."+node_data['name']
        else:
            # This is a function
            retrieved_name = node_data['name']
        
        # Get relevance score (0 if not found in ground truth)
        # Check for exact match first
        if retrieved_name in name_to_relevance and retrieved_name not in matched_gt_items:
            relevance = name_to_relevance[retrieved_name]
            matched_gt_name = retrieved_name
            
            # Store the match info
            # if tree_node:
            #     match_info = {
            #         'query_id': query_id,
            #         'query_text': query,
            #         'repo_name': repo_name,
            #         'matched_name': matched_gt_name,
            #         'retrieved_name': retrieved_name,
            #         'relevance': relevance,
            #         'rank': i + 1
            #     }
            #     match_paths.append(match_info)
            print(f"MATCH FOUND: {matched_gt_name} at rank {i+1}")
                
        else:
            # If no exact match and this is a class node, check if it contains any ground truth methods
            node_type = node_data.get('node_type', '')
            if node_type == 'Class':
                class_name = node_data['name']
                # Look for any ground truth methods that belong to this class
                for gt_name, gt_relevance in name_to_relevance.items():
                    if ('.' in gt_name and gt_name.startswith(f"{class_name}.") and 
                        gt_name not in matched_gt_items):
                        # Found a method in this class that hasn't been matched yet
                        if gt_relevance > relevance:
                            relevance = gt_relevance
                            matched_gt_name = gt_name
                            
                            # Store the class-based match info
                            # if tree_node:
                            #     match_info = {
                            #         'query_id': query_id,
                            #         'query_text': query,
                            #         'repo_name': repo_name,
                            #         'matched_name': matched_gt_name,
                            #         'retrieved_name': retrieved_name,
                            #         'relevance': relevance,
                            #         'rank': i + 1
                            #     }
                            #     match_paths.append(match_info)
                            print(f"CLASS MATCH FOUND: {matched_gt_name} via class {class_name} at rank {i+1}")
        
        # Mark the ground truth item as matched if we found one
        if matched_gt_name:
            matched_gt_items.add(matched_gt_name)
        
        retrieved_relevances.append(relevance)
    
    # Old iteration_selections storage removed - we now use the iterations_ files created above
    
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


def evaluate_retrieval_semantic(query_id, query_lookup, doc_lookup, query_to_docs, repo_name):
    """New evaluation function specifically for semantic retriever"""
    query = query_lookup[query_id][1]
    
    # Neo4j config for semantic retriever
    neo4j_config = CONFIG['neo4j']
    
    # Get results from semantic retriever
    semantic_results = semantic_main(question=query, repo_name=repo_name, neo4j_config=neo4j_config)

    # Get ground truth documents for this query
    ground_truth_docs = query_to_docs.get(query_id, [])
    
    # Create a mapping from function/method names to relevance scores
    name_to_relevance = {}
    
    for doc_info in ground_truth_docs:
        doc_id = doc_info['doc_id']
        relevance = int(doc_info['relevance'])
        
        # Get the document and extract the function name
        doc = doc_lookup[doc_id]
        func_name = doc[3]  # Function name is at index 3
        
        name_to_relevance[func_name] = relevance
    
    # Check if there are any relevant ground truth entries
    relevant_gt_count = sum(1 for rel in name_to_relevance.values() if rel > 0)
    if relevant_gt_count == 0:
        print(f"Skipping query {query_id}: No relevant ground truth entries")
        return None
    
    # Extract names from retrieved results and create relevance vector
    retrieved_relevances = []
    matched_gt_items = set()  # Track which ground truth items have been matched
    
    for i in range(min(10, len(semantic_results))):  # Only consider top 10
        relevance = 0
        matched_gt_name = None
        result = semantic_results[i]
        
        # Extract the name - use class_name field directly (no heuristics needed!)
        func_name = result.get('name', '')
        class_name = result.get('class_name')
        
        # Construct the retrieved name
        if class_name and len(class_name)>0:
            # This is a method - combine class and method name
            retrieved_name = f"{class_name}.{func_name}"
        else:
            # This is a function or global variable
            retrieved_name = func_name
        print("retrieved name: ",retrieved_name)
        # Get relevance score (0 if not found in ground truth)
        if retrieved_name in name_to_relevance and retrieved_name not in matched_gt_items:
            relevance = name_to_relevance[retrieved_name]
            matched_gt_name = retrieved_name
        
        # Mark the ground truth item as matched if we found one
        if matched_gt_name:
            matched_gt_items.add(matched_gt_name)
        
        retrieved_relevances.append(relevance)
    
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
    Process qrels list to get unique query_id, doc_id pairs with mode relevance scores.
    In case of ties, prioritize higher relevance scores.
    
    Args:
        qrels_list: List of qrel entries (can be accessed by index)
    
    Returns:
        List of deduplicated entries with mode relevance scores
    """
    # Group by (query_id, doc_id) pairs and collect relevance scores
    pairs_to_relevances = defaultdict(list)
    
    for i in range(len(qrels_list)):
        query_id = qrels_list[i][0]  # Access by index as mentioned
        doc_id = qrels_list[i][1]
        relevance = qrels_list[i][2]
        note = qrels_list[i][3] if len(qrels_list[i]) > 3 else ''
        
        pairs_to_relevances[(query_id, doc_id)].append({
            'relevance': relevance,
            'note': note,
            'original_entry': qrels_list[i]
        })
    
    # Create deduplicated results with mode relevance scores
    deduplicated_qrels = []
    
    for (query_id, doc_id), entries in pairs_to_relevances.items():
        # Get all relevance scores for this pair
        relevance_scores = [entry['relevance'] for entry in entries]
        
        # Find the mode (most common relevance score)
        # In case of ties, prioritize higher relevance scores
        relevance_counter = Counter(relevance_scores)
        
        # Get the maximum count (frequency of the most common score)
        max_count = relevance_counter.most_common(1)[0][1]
        
        # Get all scores that have the maximum count
        most_common_scores = [score for score, count in relevance_counter.items() if count == max_count]
        
        # Among the most common scores, pick the highest one
        mode_relevance = max(most_common_scores)
        
        # Create the deduplicated entry
        deduplicated_qrels.append({
            'query_id': query_id,
            'doc_id': doc_id,
            'relevance': mode_relevance,
            'count': len(entries)  # How many duplicates were merged
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
    

def main(use_semantic_retriever=False):
    with open(CONFIG['output']['input_data'], 'rb') as f:
        loaded_data = pickle.load(f)
    query_lookup = loaded_data['query_lookup']
    doc_lookup = loaded_data['doc_lookup']
    repo_mapping_sorted = loaded_data['repo_mapping_sorted']
    base_clone_dir = CONFIG['paths']['base_clone_dir']
    graph_config = CONFIG['neo4j']

    graph = Neo4jGraph(**graph_config)
    embedding_model = SentenceTransformer(CONFIG['models']['embedding'])

    # Initialize retriever (only needed for MCTS mode)
    rl_graphrag = None
    if not use_semantic_retriever:
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
    evaluation_results = {
        'retrieval_method': 'semantic' if use_semantic_retriever else 'mcts',
        'repositories': {},
        'overall_stats': {}
    }

    all_ndcg_scores = []
    all_recall_scores = []
    total_queries = 0

    for repo_name in repo_mapping_sorted.keys():

        if repo_mapping_sorted[repo_name]['qrels_count'] < CONFIG['processing']['min_qrels_count']:
            break
        print("="*60)
        print(f"Processing repository: {repo_name}")
        print(f"Using {'Semantic Retriever' if use_semantic_retriever else 'MCTS'}")
        repo_path_name=os.path.join(base_clone_dir, repo_name.replace("/", "_"))
        query_to_docs = deduplicate_qrels(repo_mapping_sorted[repo_name]['qrels'])
        
        # Get total number of nodes in the graph for this repository
        cypher_query = f"""
        MATCH (r:Repo {{name:'{repo_path_name}'}})-[:CONTAINS*]-(c)
        OPTIONAL MATCH (c)-[:HAS_METHOD*]-(m)
        RETURN count(c)+count(m)+1 as total_nodes
        """
        result = graph.query(cypher_query)
        total_graph_nodes = result[0]['total_nodes'] if result else 0
        if total_graph_nodes<=1:
            continue
        module_count=0
        cypher_query = f"""
        MATCH (r:Repo {{name:'{repo_path_name}'}})-[:CONTAINS]->(m:Module)
        RETURN count(m) as module_count
        """
        module_result = graph.query(cypher_query)
        module_count = module_result[0]['module_count'] if module_result else 100
        print("########################## ", module_result[0]['module_count']  )
        if not use_semantic_retriever and rl_graphrag:
            if module_count<20:
                rl_graphrag.original_top_k_children=max(1,module_count//2)
            else:
                rl_graphrag.original_top_k_children=min(round(module_count/10)*5,200)+15
            rl_graphrag.exploration_param=1/(8*math.sqrt(math.log(4*rl_graphrag.original_top_k_children)))
            print("########################## rl_graphrag.exploration_param: ",rl_graphrag.exploration_param)
        repo_results = {
            'repo_name': repo_name,
            'total_graph_nodes': total_graph_nodes,
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
                # Choose evaluation function based on retriever type
                if use_semantic_retriever:
                    eval_result = evaluate_retrieval_semantic(
                        query_id, query_lookup, doc_lookup, query_to_docs, repo_path_name
                    )
                else:
                    eval_result = evaluate_retrieval_mcts(
                        query_id, query_lookup, doc_lookup, query_to_docs, rl_graphrag, repo_path_name
                    )
                    
                                    
                # Skip if no relevant ground truth
                if eval_result is None:
                    print(f"  Skipped query {query_id}: No relevant ground truth")
                    continue
                
                # Count nodes in MCTS tree (only for MCTS)
                mcts_node_count = 0
                if not use_semantic_retriever and hasattr(rl_graphrag, 'root_tree_node') and rl_graphrag.root_tree_node:
                    queue = [rl_graphrag.root_tree_node]
                    while queue:
                        node = queue.pop(0)
                        mcts_node_count += 1
                        if hasattr(node, 'children'):
                            queue.extend(node.children)
                
                
                # Store query results
                query_result = {
                    'query_text': query_lookup[query_id][1],
                    'ndcg_10': eval_result['ndcg_10'],
                    'recall_10': eval_result['recall_10'],
                    'total_relevant': eval_result['total_relevant'],
                    'relevant_retrieved': eval_result['relevant_retrieved'],
                }
                
                if not use_semantic_retriever:
                    query_result['mcts_tree_nodes'] = mcts_node_count
                
                repo_results['queries'][query_id] = query_result
                
                # Collect scores for averaging
                repo_ndcg_scores.append(eval_result['ndcg_10'])
                repo_recall_scores.append(eval_result['recall_10'])
                all_ndcg_scores.append(eval_result['ndcg_10'])
                all_recall_scores.append(eval_result['recall_10'])
                total_queries += 1
                
                # Print progress
                node_info = f", MCTS nodes={mcts_node_count}" if not use_semantic_retriever else ""
                print(f"  Query {query_id}: NDCG@10={eval_result['ndcg_10']:.4f}, Recall@10={eval_result['recall_10']:.4f}{node_info}")
                
            except Exception as e:
                print(f"  Error processing query {query_id}: {str(e)}")
                continue
        
        # Calculate repository averages
        if repo_ndcg_scores:
            repo_results['repo_stats']['avg_ndcg_10'] = sum(repo_ndcg_scores) / len(repo_ndcg_scores)
            repo_results['repo_stats']['avg_recall_10'] = sum(repo_recall_scores) / len(repo_recall_scores)
        
        evaluation_results['repositories'][repo_name] = repo_results
        
        # Update overall statistics with current progress
        evaluation_results['overall_stats'] = {
            'total_repositories': len(evaluation_results['repositories']),
            'total_queries': total_queries,
            'overall_avg_ndcg_10': sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0,
            'overall_avg_recall_10': sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
        }
        
        # Save results to JSON file after each repository
        output_filename = CONFIG['output']['semantic_results'] if use_semantic_retriever else CONFIG['output']['mcts_results']
        with open(output_filename, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Repository {repo_name} completed:")
        print(f"  Avg NDCG@10: {repo_results['repo_stats']['avg_ndcg_10']:.4f}")
        print(f"  Avg Recall@10: {repo_results['repo_stats']['avg_recall_10']:.4f}")
        print(f"  Total graph nodes: {total_graph_nodes}")
        print(f"Results updated in {output_filename}")
    
    # Final update to overall statistics
    evaluation_results['overall_stats'] = {
        'total_repositories': len(repo_mapping_sorted),
        'total_queries': total_queries,
        'overall_avg_ndcg_10': sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0,
        'overall_avg_recall_10': sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
    }
    
    # Final save to JSON file
    output_filename = CONFIG['output']['semantic_results'] if use_semantic_retriever else CONFIG['output']['mcts_results']
    with open(output_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print(f"Total repositories processed: {evaluation_results['overall_stats']['total_repositories']}")
    print(f"Total queries processed: {evaluation_results['overall_stats']['total_queries']}")
    print(f"Overall average NDCG@10: {evaluation_results['overall_stats']['overall_avg_ndcg_10']:.4f}")
    print(f"Overall average Recall@10: {evaluation_results['overall_stats']['overall_avg_recall_10']:.4f}")
    print(f"Final results saved to: {output_filename}")
    
    # Close graph connection
    graph.close()
    print("Graph connection closed.")


if __name__ == "__main__":
    # Add command line argument or simple toggle
    import sys
    use_semantic = len(sys.argv) > 1 and sys.argv[1] == '--semantic'
    
    if use_semantic:
        print("Using Semantic Retriever")
    else:
        print("Using MCTS Retriever")
    
    main(use_semantic_retriever=use_semantic)