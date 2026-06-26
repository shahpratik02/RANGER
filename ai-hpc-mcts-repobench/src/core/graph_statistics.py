#!/usr/bin/env python3
"""
RepoBench Graph Statistics Collection Script
Processes each repository in RepoBench dataset by:
1. Cloning the repository at the correct commit
2. Creating a graph representation
3. Collecting and saving graph statistics
4. Moving to the next repository
"""

import os
import json
import shutil
import sys
import subprocess
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from langchain_neo4j import Neo4jGraph
import argparse

# Import functions from the other files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path
from src.utils.config import CONFIG


def clone_repo_at_commit(repo_name, created_at, base_dir=None):
    """
    Clone a repository and checkout to the latest commit before created_at
    Returns: (repo_path, commit_id, error_message)
    """
    if base_dir is None:
        base_dir = CONFIG['paths']['repobench_base']
    
    try:
        os.makedirs(base_dir, exist_ok=True)
        clean_repo_name = repo_name.replace("/", "_")
        repo_path = os.path.join(base_dir, clean_repo_name)

        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        print(f"  Cloning {repo_name}...")
        clone_url = f"https://github.com/{repo_name}.git"

        env = os.environ.copy()
        env.update({
            'GIT_TERMINAL_PROMPT': '0',
            'GIT_ASKPASS': 'echo',
            'SSH_ASKPASS': 'echo',
        })

        try:
            result = subprocess.run(
                ["git", "clone", clone_url, repo_path], 
                capture_output=True, 
                text=True,
                timeout=CONFIG['processing']['clone_timeout'],
                env=env
            )
        except subprocess.TimeoutExpired:
            return None, None, f"Repository {repo_name} clone timed out"

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "repository not found" in error_msg.lower():
                return None, None, f"Repository {repo_name} not found (404)"
            elif "permission denied" in error_msg.lower():
                return None, None, f"Repository {repo_name} is private (403)"
            else:
                return None, None, f"Git clone failed for {repo_name}: {error_msg}"

        os.chdir(repo_path)
        
        # Use fixed date for consistency
        created_date = datetime(2023, 12, 31, 23, 59, 59)

        print(f"  Finding latest commit before {created_date}...")
        result = subprocess.run(
            ["git", "rev-list", "-n", "1", f"--before={created_date.isoformat()}", "HEAD"],
            capture_output=True, text=True
        )

        target_commit = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "HEAD"

        print(f"  Checking out to commit: {target_commit[:8] if target_commit != 'HEAD' else 'HEAD'}...")
        subprocess.run(["git", "checkout", target_commit], capture_output=True, text=True)

        # Get the actual commit hash after checkout
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            actual_commit = result.stdout.strip()
        else:
            actual_commit = target_commit

        print(f"  ✓ Successfully prepared repository at {repo_path}")
        return repo_path, actual_commit, None

    except Exception as e:
        return None, None, f"Unexpected error cloning repository {repo_name}: {e}"


def find_requirements_file(repo_path):
    """Find a requirements file in the repository"""
    possible_files = [
        "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
        "test_requirements.txt", "dev-requirements.txt", "requirements/base.txt",
        "requirements/dev.txt", "requirements/test.txt"
    ]

    for filename in possible_files:
        filepath = os.path.join(repo_path, filename)
        if os.path.exists(filepath):
            return filepath

    # Create a dummy requirements file if none found
    dummy_requirements = os.path.join(repo_path, "dummy_requirements.txt")
    with open(dummy_requirements, "w") as f:
        f.write("# Dummy requirements file\n")
    return dummy_requirements


def create_graph_subprocess(repo_path, requirements_path, neo4j_config, graph):
    """Run generate_graph.py as a subprocess with arguments"""
    try:
        # Clear database before creating graph
        clear_neo4j_database(graph)
        
        venv_python = CONFIG['paths']['venv_python']
        script_path = CONFIG['paths']['generate_graph_script']
        
        cmd = [
            venv_python, script_path,
            "--root-dir", repo_path,
            "--requirements-path", requirements_path,
            "--url", neo4j_config["url"],
            "--username", neo4j_config["username"],
            "--password", neo4j_config["password"],
        ]

        result = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, 
            timeout=CONFIG['processing']['graph_generation_timeout']
        )

        if result.returncode == 0:
            return True, None
        else:
            return False, f"Graph creation failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"Graph creation timed out after {CONFIG['processing']['graph_generation_timeout']} seconds"
    except Exception as e:
        return False, f"Error running graph creation: {e}"


def clear_neo4j_database(graph):
    """Clear Neo4j database"""
    try:
        graph.query("MATCH (n) DETACH DELETE (n)")
        print("  ✓ Database cleared")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not clear database: {e}")


def count_python_files(repo_path):
    """Count the total number of Python files in the repository"""
    python_file_count = 0
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                python_file_count += 1
    return python_file_count


def save_graph_statistics(repo_name, repo_path, commit_id, graph):
    """Save graph statistics to JSON file"""
    try:
        # Query to get node counts by type
        node_count_query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY node_type
        """
        
        result = graph.query(node_count_query)
        
        # Initialize statistics
        stats = {
            "repo_name": repo_name,
            "commit_id": commit_id,
            "short_commit": commit_id[:8] if commit_id and commit_id != "HEAD" else "HEAD",
            "total_nodes": 0,
            "node_counts": {
                "Module": 0,
                "Class": 0, 
                "Function": 0,
                "Method": 0,
                "GlobalVariable": 0,
                "Field": 0,
                "Repo": 0
            },
            "python_files": count_python_files(repo_path),
            "timestamp": datetime.now().isoformat()
        }
        
        # Process results
        for record in result:
            node_type = record["node_type"]
            count = record["count"]
            stats["total_nodes"] += count
            
            if node_type in stats["node_counts"]:
                stats["node_counts"][node_type] = count
        
        # Create repository key with commit ID
        short_commit = commit_id[:8] if commit_id and commit_id != "HEAD" else "HEAD"
        repo_key = f"{repo_name}@{short_commit}"
        
        # Load existing data
        filename = "repobench_graph_statistics.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = {}
        
        # Add new repository statistics
        all_stats[repo_key] = stats
        
        # Save updated statistics
        with open(filename, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"✓ Graph statistics saved for {repo_key}")
        print(f"  Total nodes: {stats['total_nodes']}, Python files: {stats['python_files']}")
        
        return stats
        
    except Exception as e:
        print(f"Error saving graph statistics for {repo_name}@{commit_id[:8] if commit_id else 'HEAD'}: {e}")
        return None


def process_repository(repo_name, examples, neo4j_config, graph):
    """Process a single repository: clone, create graph, collect statistics"""
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"Number of examples: {len(examples)}")
    print(f"{'='*60}")
    
    original_cwd = os.getcwd()
    try:
        # Get created_at from the first example
        first_example = examples[0]
        created_at = first_example.get("created_at", "2023-01-01T00:00:00Z")

        # Clone repository at the correct commit
        repo_path, commit_id, error_msg = clone_repo_at_commit(repo_name, created_at)
        if not repo_path:
            print(f"Skipping {repo_name}: {error_msg}")
            return None, "skipped"

        print(f"Using commit: {commit_id[:8] if commit_id and commit_id != 'HEAD' else 'HEAD'}")

        # Find requirements file
        requirements_path = find_requirements_file(repo_path)
        print(f"Using requirements file: {requirements_path}")

        # Create graph representation
        print(f"Creating graph for {repo_name}...")
        success, error = create_graph_subprocess(
            repo_path, requirements_path, neo4j_config, graph
        )
        
        if not success:
            print(f"Graph creation failed: {error}")
            return None, "failed"
        
        print("✓ Graph created successfully")

        # Collect and save graph statistics
        print(f"Collecting graph statistics for {repo_name}...")
        stats = save_graph_statistics(repo_name, repo_path, commit_id, graph)
        
        if stats:
            return stats, "success"
        else:
            return None, "failed"

    except Exception as e:
        print(f"Error processing repository {repo_name}: {e}")
        return None, "failed"
    finally:
        # Always return to original directory
        os.chdir(original_cwd)


def main():
    """Main function to process all RepoBench repositories and collect statistics"""
    parser = argparse.ArgumentParser(description="RepoBench Graph Statistics Collection Script")
    parser.add_argument(
        "--start-from",
        type=str,
        help="Repository name to start processing from (skip previous ones)"
    )
    args = parser.parse_args()

    # Use configuration
    neo4j_config = CONFIG['database']['neo4j']
    graph = Neo4jGraph(
        url=neo4j_config['url'], 
        username=neo4j_config['username'], 
        password=neo4j_config['password']
    )

    try:
        # Load RepoBench dataset
        print("Loading RepoBench dataset...")
        dataset_config = CONFIG['dataset']['repobench']
        dataset = load_dataset(dataset_config['name'], split=dataset_config['split'])
        dataset_list = list(dataset)

        # Group by repository
        print("Grouping by repository...")
        grouped = defaultdict(list)
        for item in dataset_list:
            repo_name = item["repo_name"]
            grouped[repo_name].append(item)

        # Sort repositories by number of examples (descending) and filter
        sorted_groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_groups = [item for item in sorted_groups if len(item[1]) >= 5]
        
        print(f"Dataset contains {len(dataset_list)} examples from {len(grouped)} repositories")
        print(f"After filtering (>=5 examples): {len(sorted_groups)} repositories")

        # Skip to start_from repository if specified
        if args.start_from:
            start_idx = None
            for i, (repo_name, _) in enumerate(sorted_groups):
                if repo_name == args.start_from:
                    start_idx = i
                    break
            if start_idx is not None:
                sorted_groups = sorted_groups[start_idx:]
                print(f"Starting from repository: {args.start_from}")
            else:
                print(f"Warning: Start repository '{args.start_from}' not found")

        # Load existing statistics to skip already processed repositories
        existing_stats = {}
        stats_filename = "repobench_graph_statistics.json"
        if os.path.exists(stats_filename):
            try:
                with open(stats_filename, 'r') as f:
                    existing_stats = json.load(f)
                print(f"Loaded existing statistics for {len(existing_stats)} repositories")
            except (json.JSONDecodeError, FileNotFoundError):
                existing_stats = {}
                print("Could not load existing statistics, starting fresh")
        else:
            print("No existing statistics file found, processing all repositories")

        # Process each repository
        successful_repos = 0
        failed_repos = 0
        skipped_repos = 0
        already_processed = 0
        
        original_cwd = os.getcwd()

        try:
            for repo_name, examples in tqdm(sorted_groups, desc="Processing repositories"):
                try:
                    os.chdir(original_cwd)
                    
                    # Check if this repository is already processed
                    # We need to check with different possible commit IDs since we don't know the exact commit yet
                    repo_already_processed = False
                    for existing_key in existing_stats.keys():
                        if existing_key.startswith(f"{repo_name}@"):
                            print(f"⏭️  {repo_name} already processed (found {existing_key}), skipping...")
                            already_processed += 1
                            repo_already_processed = True
                            break
                    
                    if repo_already_processed:
                        continue
                    
                    stats, status = process_repository(repo_name, examples, neo4j_config, graph)

                    if status == "success":
                        successful_repos += 1
                        print(f"✅ {repo_name} processed successfully")
                    elif status == "skipped":
                        skipped_repos += 1
                        print(f"⏭️  {repo_name} skipped")
                    else:  # failed
                        failed_repos += 1
                        print(f"❌ {repo_name} failed")

                except KeyboardInterrupt:
                    print("\n\nProcessing interrupted by user")
                    break
                except Exception as e:
                    print(f"✗ Unexpected error with {repo_name}: {e}")
                    failed_repos += 1
                    continue

        finally:
            os.chdir(original_cwd)

        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total repositories in dataset: {len(sorted_groups)}")
        print(f"Already processed (skipped): {already_processed}")
        print(f"Successfully processed: {successful_repos}")
        print(f"Skipped (not found/private): {skipped_repos}")
        print(f"Failed (other errors): {failed_repos}")
        print(f"\n✓ Graph statistics saved to: repobench_graph_statistics.json")

    finally:
        # Always close the graph connection
        print("Closing graph connection...")
        try:
            graph.close()
            print("✓ Graph connection closed successfully")
        except Exception as e:
            print(f"⚠️ Warning: Error closing graph connection: {e}")


if __name__ == "__main__":
    main()