import ir_datasets
from collections import defaultdict
import re
import subprocess
import os
import sys
from urllib.parse import urlparse
from langchain_neo4j import Neo4jGraph
import torch
# sys.path.append('/home/ec2-user/to_push')  # Removed - using proper imports now

from src.core.generate_graph import main as generate_graph_main
from src.data.CodeSearchNet_dataprep import GraphCorpusFilter
from src.utils.simple_config import CONFIG
# Add the GitHub URL parsing functions
def parse_github_url(github_url):
    """
    Parse a GitHub URL to extract repository info and commit ID.
    """
    parsed = urlparse(github_url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) < 4 or path_parts[2] != 'blob':
        raise ValueError("Invalid GitHub URL format")
    
    owner = path_parts[0]
    repo = path_parts[1]
    commit_id = path_parts[3]
    file_path = '/'.join(path_parts[4:]) if len(path_parts) > 4 else ''
    
    return {
        'owner': owner,
        'repo': repo,
        'commit_id': commit_id,
        'file_path': file_path,
        'repo_url': f"https://github.com/{owner}/{repo}.git"
    }
# Add the graph checking function
def check_repository_graph_state(repo_name, url, username, password):
    """
    Check the state of a repository graph and return the appropriate action.
    
    Returns:
        - 'skip': Repository fully embedded, skip processing
        - 'delete': Repository exists but no embeddings, delete all nodes
        - 'process': Repository doesn't exist or partially embedded, continue processing
    """
    
    try:
        graph = Neo4jGraph(url=url, username=username, password=password)
        
        # Check if repository node exists
        repo_check_query = """
        MATCH (r:Repo {name: $repo_name})
        RETURN r
        """
        repo_results = graph.query(repo_check_query, {"repo_name": repo_name})
        
        if not repo_results:
            print(f"Repository {repo_name} not found in graph - will process")
            graph.close()
            return 'process'
        
        # Repository exists, check module embedding status
        module_stats_query = """
        MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)
        WITH count(m) as total_modules, 
             count(CASE WHEN m.embedding IS NOT NULL THEN 1 END) as embedded_modules
        RETURN total_modules, embedded_modules
        """
        stats_results = graph.query(module_stats_query, {"repo_name": repo_name})
        
        if not stats_results or stats_results[0]['total_modules'] == 0:
            print(f"Repository {repo_name} has no modules - will delete")
            graph.close()
            return 'delete'
        
        total_modules = stats_results[0]['total_modules']
        embedded_modules = stats_results[0]['embedded_modules']
        
        print(f"Repository {repo_name}: {embedded_modules}/{total_modules} modules have embeddings")
        
        if embedded_modules == 0:
            print(f"Repository {repo_name} has no embeddings - will delete and reprocess")
            graph.close()
            return 'delete'
        elif embedded_modules == total_modules:
            print(f"Repository {repo_name} is fully embedded - skipping")
            graph.close()
            return 'skip'
        else:
            print(f"Repository {repo_name} is partially embedded - will continue processing")
            graph.close()
            return 'process'
            
    except Exception as e:
        print(f"Error checking repository {repo_name}: {e}")
        return 'process'

def delete_repository_nodes(repo_name, url, username, password):
    """
    Delete all nodes for a repository including traversing CONTAINS and HAS_METHOD edges.
    """
    
    try:
        graph = Neo4jGraph(url=url, username=username, password=password)
        
        # Delete all nodes related to the repository
        delete_query = """
        MATCH (r:Repo {name: $repo_name})
        OPTIONAL MATCH (r)-[:CONTAINS]->(m:Module)
        OPTIONAL MATCH (m)-[:CONTAINS]->(child)
        OPTIONAL MATCH (child)-[:HAS_METHOD]->(method)
        DETACH DELETE r, m, child, method
        """
        
        graph.query(delete_query, {"repo_name": repo_name})
        print(f"Deleted all nodes for repository: {repo_name}")
        graph.close()
        
    except Exception as e:
        print(f"Error deleting nodes for repository {repo_name}: {e}")

def clone_and_checkout(repo_info, target_dir):
    """
    Clone the repository and checkout the specific commit.
    """
    repo_url = repo_info['repo_url']
    commit_id = repo_info['commit_id']
    
    try:
        # Remove existing directory if it exists
        if os.path.exists(target_dir):
            print(f"Removing existing directory: {target_dir}")
            subprocess.run(['rm', '-rf', target_dir], check=True)
        
        # Clone the repository
        print(f"Cloning repository: {repo_url} to {target_dir}")
        subprocess.run([
            'git', 'clone', repo_url, target_dir
        ], check=True, capture_output=True, text=True)
        
        # Checkout the specific commit
        print(f"Checking out commit: {commit_id}")
        subprocess.run([
            'git', 'checkout', commit_id
        ], cwd=target_dir, check=True, capture_output=True, text=True)
        
        print(f"Successfully cloned and checked out commit {commit_id}")
        return os.path.abspath(target_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

def find_requirements_file(repo_path):
    """
    Find a requirements file in the repository.
    """
    possible_files = [
        'requirements.txt',
        'test_requirements.txt',
        'dev_requirements.txt',
        'requirements-dev.txt',
        'requirements/base.txt',
        'requirements/requirements.txt'
    ]
    
    for req_file in possible_files:
        full_path = os.path.join(repo_path, req_file)
        if os.path.exists(full_path):
            return full_path
    
    # Create a dummy requirements file if none found
    dummy_req_path = os.path.join(repo_path, 'requirements.txt')
    with open(dummy_req_path, 'w') as f:
        f.write("# Auto-generated dummy requirements file\n")
    
    return dummy_req_path

def main():
    # Load datasets
    print("Loading CodeSearchNet datasets...")
    dataset = ir_datasets.load("codesearchnet/challenge")
    entire_corpus = ir_datasets.load("codesearchnet")
    
    print("Building doc_id to repo mapping for Python functions...")
    doc_to_repo = {}
    python_doc_ids = set()
    
    for doc in dataset.docs_iter():
        if doc.language == 'python':
            doc_to_repo[doc.doc_id] = doc.repo
            python_doc_ids.add(doc.doc_id)
    
    print(f"Found {len(python_doc_ids)} Python functions")
    
    # Group qrels by repository for Python functions only
    print("Grouping qrels by repository...")
    qrels_by_repo = defaultdict(list)
    
    for qrel in dataset.qrels_iter():
        if qrel.doc_id in python_doc_ids:
            repo = doc_to_repo[qrel.doc_id]
            qrels_by_repo[repo].append(qrel)
    
    sorted_repos = sorted(qrels_by_repo.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Build lookups
    query_lookup = {}
    doc_lookup = {}
    
    for query in dataset.queries_iter():
        query_lookup[query.query_id] = query
    
    for doc in dataset.docs_iter():
        if doc.language == 'python':
            doc_lookup[doc.doc_id] = doc
    
    # Process entire corpus
    print("Building doc_id to repo mapping for entire corpus...")
    entire_corpus_doc_to_repo = {}
    entire_corpus_python_doc_ids = set()
    
    for doc in entire_corpus.docs_iter():
        if doc.language == 'python':
            entire_corpus_doc_to_repo[doc.doc_id] = doc.repo
            entire_corpus_python_doc_ids.add(doc.doc_id)
    
    print(f"Found {len(entire_corpus_python_doc_ids)} Python functions in entire corpus")
    
    # Group documents by repositories
    print("Grouping documents by repository...")
    docs_by_repo = defaultdict(list)
    
    for doc in entire_corpus.docs_iter():
        if doc.language == 'python':
            docs_by_repo[doc.repo].append(doc)
    
    print(f"Found {len(docs_by_repo)} repositories")
    
    # Sort repositories by number of documents
    sorted_repos_by_doc_count = sorted(docs_by_repo.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Display top 10 repositories by document count
    print("\nTop 10 repositories by document count:")
    for i, (repo, docs) in enumerate(sorted_repos_by_doc_count[:10]):
        print(f"{i+1}. {repo}: {len(docs)} documents")
    
    # Convert sorted_repos to dictionary
    sorted_repos_dict = {}
    for repo_name, qrels_list in sorted_repos:
        sorted_repos_dict[repo_name] = qrels_list
    
    # Create repo mapping
    repo_mapping = {}
    for repo_name, docs_list in sorted_repos_by_doc_count:
        if repo_name in sorted_repos_dict and len(sorted_repos_dict[repo_name]) > 0:
            repo_mapping[repo_name] = {
                'documents': docs_list,
                'qrels': sorted_repos_dict[repo_name],
                'doc_count': len(docs_list),
                'qrels_count': len(sorted_repos_dict[repo_name])
            }
    
    print(f"Found {len(repo_mapping)} repositories with both documents and qrels")
    repo_mapping_sorted = dict(sorted(repo_mapping.items(), key=lambda item: item[1]['qrels_count'], reverse=True))
    # Import required modules

    
    # Neo4j connection details from config
    neo4j_config = CONFIG['neo4j']
    url = neo4j_config['url']
    username = neo4j_config['username']
    password = neo4j_config['password']
    
    # Process repositories
    base_clone_dir = CONFIG['paths']['base_clone_dir']
    os.makedirs(base_clone_dir, exist_ok=True)
    
    # for i in range(1,len(sorted_repos_by_doc_count)):
    #     repo_name = sorted_repos_by_doc_count[i+4][0]

    i=0
    for x in repo_mapping_sorted:
        i+=1
        repo_name = x
        if repo_name not in repo_mapping:
            print(f"Skipping {repo_name} - no qrels found")
            continue
        if repo_mapping_sorted[repo_name]['qrels_count'] < CONFIG['processing']['min_qrels_count']:
            break
        print(f"\n{'='*60}")
        print(f"Processing repository {i+1}/10: {repo_name}")
        print(f"{'='*60}")
                # Check repository graph state
        repo_path_name=os.path.join(base_clone_dir, repo_name.replace("/", "_"))
        graph_state = check_repository_graph_state(repo_path_name, url, username, password)
        
        if graph_state == 'skip':
            print(f"Repository {repo_name} already fully embedded - skipping")
            continue
        elif graph_state == 'delete':
            print(f"Deleting existing nodes for repository {repo_name}")
            delete_repository_nodes(repo_path_name, url, username, password)

        try:
            # Get GitHub URL from qrels
            qrels_list = repo_mapping_sorted[repo_name]['qrels']
            if not qrels_list:
                print(f"No qrels found for {repo_name}")
                continue
            
            # Get the first qrel's document
            first_qrel = qrels_list[0]
            doc_id = first_qrel.doc_id
            
            # Find the document in doc_lookup
            if doc_id not in doc_lookup:
                print(f"Document {doc_id} not found in doc_lookup")
                continue
                
            doc = doc_lookup[doc_id]
            github_url = doc_id
            
            print(f"GitHub URL: {github_url}")
            
            # Parse GitHub URL
            repo_info = parse_github_url(github_url)
            print(f"Parsed: {repo_info['owner']}/{repo_info['repo']} @ {repo_info['commit_id']}")
            
            # Set up clone directory
            clone_dir = os.path.join(base_clone_dir, f"{repo_info['owner']}_{repo_info['repo']}")
            
            # Clone repository
            repo_path = clone_and_checkout(repo_info, clone_dir)
            
            # Find requirements file
            requirements_path = find_requirements_file(repo_path)
            print(f"Using requirements file: {requirements_path}")
            
            # Generate graph
            print("Generating graph...")
            generate_graph_main(repo_path, requirements_path, url, username, password)
            
            # Prune graph
            print("Pruning graph...")
            filter_tool = GraphCorpusFilter(url, username, password)
            filter_tool.filter_graph(repo_mapping_sorted[repo_name]['documents'], repo_path_name)
            filter_tool.graph.close()
            
            # Generate embeddings
            print("Generating embeddings...")
            subprocess.run([
                'python', '-m', 'src.core.add_embeddings_final',
                repo_path,
            ], cwd=CONFIG['paths']['project_root'], check=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Successfully processed repository: {repo_name}")

            
        except Exception as e:
            print(f"Error processing repository {repo_name}: {e}")
            import traceback
            traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    
    print("\n" + "="*60)
    print(f"Finished processing {i} repositories!")
    print("="*60)

if __name__ == "__main__":
    main()