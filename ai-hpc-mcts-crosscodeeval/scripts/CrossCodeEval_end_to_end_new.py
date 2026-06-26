import os
import json
import shutil
import sys
import subprocess
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from langchain_neo4j import Neo4jGraph
from transformers import AutoTokenizer, AutoModel
import torch
import argparse

from src.evaluation.CrossCodeEvaluation import (
    evaluate_retrieval,
    save_results,
    PythonCodeBM25Searcher,
)
from src.utils.config_loader import CONFIG


def clone_repo_at_commit(repo_key, base_dir=CONFIG["paths"]["repo_base_dir"]):
    """
    Parse a repository key like 'turboderp-exllama-a544085', clone the repo, and checkout to the specified commit.

    Args:
        repo_key: String like 'turboderp-exllama-a544085'
        base_dir: Base directory to store cloned repos

    Returns:
        tuple: (repo_path, error_message) where repo_path is None if failed
    """
    # Parse repo name and commit id
    parts = repo_key.split("-")
    if len(parts) < 3:
        return None, f"Invalid repository key format: {repo_key}"
    commit_id = parts[-1]

    # Generate all possible owner/repo combinations
    repo_name_options = []
    name_parts = parts[:-1]  # Exclude commit id

    # Try all possible splits: owner can be 1 to n-1 parts, repo gets the rest
    for i in range(1, len(name_parts)):
        owner = "-".join(name_parts[:i])
        repo = "-".join(name_parts[i:])
        repo_name_options.append(f"{owner}/{repo}")

    print(f"  Trying repository name options: {repo_name_options}")

    # Try each repository name option
    for repo_name in repo_name_options:
        try:
            os.makedirs(base_dir, exist_ok=True)
            clean_repo_name = repo_name.replace("/", "_")
            repo_path = os.path.join(base_dir, clean_repo_name)
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)

            print(f"  Trying to clone {repo_name}...")
            clone_url = f"https://github.com/{repo_name}.git"
            env = os.environ.copy()
            env.update(
                {
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_ASKPASS": "echo",
                    "SSH_ASKPASS": "echo",
                }
            )
            try:
                result = subprocess.run(
                    ["git", "clone", clone_url, repo_path],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                print(f"    Timeout for {repo_name}, trying next option...")
                continue

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                if (
                    "repository not found" in error_msg.lower()
                    or "not found" in error_msg.lower()
                ):
                    print(
                        f"    Repository {repo_name} not found, trying next option..."
                    )
                    continue
                elif (
                    "permission denied" in error_msg.lower()
                    or "forbidden" in error_msg.lower()
                ):
                    print(
                        f"    Repository {repo_name} is private, trying next option..."
                    )
                    continue
                elif "could not resolve host" in error_msg.lower():
                    print(f"    Network error for {repo_name}, trying next option...")
                    continue
                else:
                    print(
                        f"    Git clone failed for {repo_name}: {error_msg}, trying next option..."
                    )
                    continue

            # If we get here, clone was successful
            os.chdir(repo_path)
            print(f"  Checking out to commit: {commit_id}...")
            result = subprocess.run(
                ["git", "checkout", commit_id], capture_output=True, text=True
            )
            if result.returncode != 0:
                error_msg = (
                    result.stderr.strip() if result.stderr else "Unknown checkout error"
                )
                print(
                    f"    Git checkout failed for commit {commit_id}: {error_msg}, trying next option..."
                )
                # Clean up failed repo
                os.chdir("/")
                shutil.rmtree(repo_path)
                continue

            print(f"  ✓ Successfully prepared repository at {repo_path}")
            return repo_path, None

        except Exception as e:
            print(f"    Unexpected error with {repo_name}: {e}, trying next option...")
            continue

    # If we get here, all options failed
    return (
        None,
        f"All repository name options failed for {repo_key}. Tried: {repo_name_options}",
    )


def find_requirements_file(repo_path):
    """
    Find a requirements file in the repository

    Args:
        repo_path: Path to the repository

    Returns:
        str: Path to requirements file or None if not found
    """
    possible_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "test_requirements.txt",
        "dev-requirements.txt",
        "requirements/base.txt",
        "requirements/dev.txt",
        "requirements/test.txt",
        # "pyproject.toml",
        # "setup.py",
    ]

    for filename in possible_files:
        filepath = os.path.join(repo_path, filename)
        if os.path.exists(filepath):
            return filepath

    # Create a dummy requirements file if none found
    dummy_requirements = os.path.join(repo_path, CONFIG["output"]["dummy_requirements"])
    with open(dummy_requirements, "w") as f:
        f.write("# Dummy requirements file\n")

    return dummy_requirements


def save_results_tracking(
    repo_name,
    exact_match_rate,
    average_edit_similarity,
    total_examples,
    filename=CONFIG["output"]["results_tracking_file"],
):
    """
    Save or update the results tracking JSON file after each repository

    Args:
        repo_name: Name of the repository
        exact_match_rate: Exact Match rate (0.0-1.0)
        average_edit_similarity: Average Edit Similarity score (0.0-1.0)
        total_examples: Total number of examples processed
        filename: Name of the JSON file to save to
    """
    tracking_data = {}

    # Load existing data if file exists
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                tracking_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            tracking_data = {}

    # Add new repository data
    tracking_data[repo_name] = {
        "exact_match_rate": exact_match_rate,
        "average_edit_similarity": average_edit_similarity,
        "total_examples": total_examples,
        "processed_at": datetime.now().isoformat(),
    }

    # Save updated data
    with open(filename, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(
        f"✓ Updated results tracking: {repo_name} - EM: {exact_match_rate:.3f}, ES: {average_edit_similarity:.3f}"
    )


def create_graph_subprocess(repo_path, requirements_path, neo4j_config, graph):
    """Run create_graph as a subprocess for better performance"""
    """Run generate_graph.py as a subprocess with arguments"""
    try:
        # Clear database before creating graph
        clear_neo4j_database(graph)
        venv_python = CONFIG["paths"]["venv_python"]
        script_path = CONFIG["paths"]["generate_graph_script"]
        # Build command with arguments
        cmd = [
            venv_python,
            script_path,
            "--root-dir",
            repo_path,
            "--requirements-path",
            requirements_path,
            "--url",
            neo4j_config["url"],
            "--username",
            neo4j_config["username"],
            "--password",
            neo4j_config["password"],
        ]

        # Run subprocess with timeout
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=CONFIG["processing"]["timeout_seconds"],
        )

        if result.returncode == 0:
            return True, None
        else:
            return False, f"Graph creation failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return (
            False,
            f"Graph creation timed out after {CONFIG['processing']['timeout_seconds']} seconds",
        )
    except Exception as e:
        return False, f"Error running graph creation: {e}"


def clear_neo4j_database(graph):
    """Clear Neo4j database before processing"""
    try:
        graph.query("MATCH (n) DETACH DELETE (n)")
        print("  ✓ Database cleared")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not clear database: {e}")


def process_repository(
    repo_key, examples, neo4j_config, graph, model, tokenizer, retriever_type
):
    """
    Process a single repository: clone, create graph, evaluate

    Args:
        repo_key: Repository key like 'turboderp-exllama-a544085'
        examples: List of examples for this repository
        graph: Neo4jGraph instance

    Returns:
        tuple: (results, status) where status is 'success', 'skipped', or 'failed'
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_key}")
    print(f"Number of examples: {len(examples)}")
    print(f"{'='*60}")
    print("Clearing Neo4j database...")
    clear_neo4j_database(graph)
    original_cwd = os.getcwd()
    # try:
    # Parse repo name and commit id from repo_key
    parts = repo_key.split("-")
    if len(parts) < 3:
        print(f"Invalid repository key format: {repo_key}")
        return None, "skipped"
    commit_id = parts[-1]
    repo_name = f"{parts[0]}/{ '-'.join(parts[1:-1]) }"

    print(f"Repository commit id: {commit_id}")

    # Clone repository at the correct commit
    repo_path, error_msg = clone_repo_at_commit(repo_key)
    if not repo_path:
        print(f"Skipping {repo_key}: {error_msg}")
        try:
            with open(CONFIG["output"]["skipped_repos_file"], "a") as f:
                f.write(f"{repo_key}\n")
        except Exception as e:
            print(
                f"Warning: Could not write to {CONFIG['output']['skipped_repos_file']}: {e}"
            )
        return None, "skipped"

    # Find requirements file
    requirements_path = find_requirements_file(repo_path)
    print(f"Using requirements file: {requirements_path}")

    # Create graph representation
    print(f"Creating graph for {repo_key}...")
    try:
        success, error = create_graph_subprocess(
            repo_path, requirements_path, neo4j_config, graph
        )
        if success:
            print(f" Graph created successfully")
        else:
            print(f" Graph creation failed: {error}")
            return None, "failed"
    except Exception as e:
        print(f" Graph creation failed: {e}")
        return None, "failed"

    # Build BM25 index if needed
    if retriever_type in ["bm25", "hybrid"]:
        print("Building BM25 index...")
        bm25_searcher = PythonCodeBM25Searcher(repo_path)
    else:
        bm25_searcher = None
    # If using semantic retriever, build the index now
    # if retriever_type == "semantic":
    #     print("Building semantic index...")
    #     try:
    #         try:
    #             CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
    #         except Exception:
    #             pass
    #         collection = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
    #         success = build_index_from_graph(repo_key, collection, neo4j_config)
    #         if success:
    #             print("  ✓ Semantic index built successfully.")
    #         else:
    #             print("  ✗ Semantic index building failed.")
    #             return None, "failed"
    #     except Exception as e:
    #         print(f"  ✗ Error building semantic index: {e}")
    #         return None, "failed"

    # Evaluate retrieval performance
    print(f"Evaluating code generation for {repo_key}...")

    query_format = """
Given repo_name:{repo_name}
Given file_name:{file_name}
Fetch the most important connected nodes from the graph to predict the next line of the below code
{code}"""

    repo_dataset = examples

    # try:
    results = evaluate_retrieval(
        repo_path=repo_path,
        dataset=repo_dataset,
        query_format=query_format,
        repo_filter=repo_key,
        retriever_type=retriever_type,
        repo_name_for_semantic=repo_key,
        model=model,
        tokenizer=tokenizer,
        neo4j_config=neo4j_config,
        graph=graph,
        bm25_searcher=bm25_searcher,
    )
    if bm25_searcher:
        bm25_searcher.clear_index()
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Total Examples: {results['summary']['total_examples']}")
    print(f"Exact Matches: {results['summary']['exact_matches']}")
    print(f"EM Rate: {results['summary']['em_rate']:.3f}")
    print(
        f"Average Edit Similarity: {results['summary']['average_edit_similarity']:.3f}"
    )
    print(f"Max Edit Similarity: {results['summary']['max_edit_similarity']:.3f}")
    print(f"Min Edit Similarity: {results['summary']['min_edit_similarity']:.3f}")

    safe_repo_name = repo_key.replace("/", "_")
    results_path = os.path.join(original_cwd, f"cceval_{safe_repo_name}_results.csv")

    save_results(
        results,
        f"{original_cwd}/{CONFIG['output']['results_dir']}cceval_{safe_repo_name}",
    )

    return results, "success"

    #     except Exception as e:
    #         print(f" Evaluation failed: {e}")
    #         return None, "failed"

    # except Exception as e:
    #     print(f"Error processing repository {repo_key}: {e}")
    #     return None, "failed"


def load_jsonl(filename):
    """Load JSONL file - each line is a separate JSON object"""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Each line is a complete JSON object
            data.append(json.loads(line.strip()))
    return data


def main():
    """Main function to process all CrossCodeEval repositories"""
    parser = argparse.ArgumentParser(description="CrossCodeEval Processing Script")
    # Create mutually exclusive group for retriever types
    retriever_group = parser.add_mutually_exclusive_group()
    retriever_group.add_argument(
        "--hybrid", action="store_true", help="Use graph+BM25 hybrid retriever"
    )
    retriever_group.add_argument(
        "--bm25", action="store_true", help="Use only BM25 retriever"
    )
    retriever_group.add_argument(
        "--graph", action="store_true", help="Use only graph retriever"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=CONFIG["paths"]["data_path"],
        help="Path to CrossCodeEval JSONL data file",
    )
    args = parser.parse_args()

    # Determine retriever type from flags
    if args.hybrid:
        retriever_type = "hybrid"
    elif args.bm25:
        retriever_type = "bm25"
    elif args.graph:
        retriever_type = "graph"
    else:
        # Default to graph if no flag specified
        retriever_type = "graph"

    print(f"Using retriever: {retriever_type}")

    # Neo4j configuration
    neo4j_config = {
        "url": CONFIG["database"]["url"],
        "username": CONFIG["database"]["username"],
        "password": CONFIG["database"]["password"],
    }
    graph = Neo4jGraph(
        url=neo4j_config["url"],
        username=neo4j_config["username"],
        password=neo4j_config["password"],
    )
    # with open("cceval_results_tracking_hybrid_deepseek_6_7b_final_2.json", "r") as f:
    #     data = json.load(f)
    # repo_names_processed = []
    # for repo_name, metrics in data.items():
    #     repo_names_processed.append(repo_name)

    try:
        # Load embedding model
        EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
        print(f"Loading {EMBEDDING_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side="left")
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, torch_dtype=torch.float16, trust_remote_code=True
        )

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()

        # Load CrossCodeEval dataset
        print("Loading CrossCodeEval dataset...")
        actual_data = load_jsonl(args.data_path)

        # Group by repository
        print("Grouping by repository...")
        repo_groups = defaultdict(list)
        for item in actual_data:
            repo = item["metadata"]["repository"]
            repo_groups[repo].append(item)

        # Sort repositories by number of items (descending)
        sorted_repos = sorted(
            repo_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        sorted_repos = [
            item
            for item in sorted_repos
            if len(item[1]) >= CONFIG["processing"]["min_examples_per_repo"]
        ]

        print(
            f"Dataset contains {len(actual_data)} examples from {len(repo_groups)} repositories"
        )
        print(
            f"After filtering (>={CONFIG['processing']['min_examples_per_repo']} examples): {len(sorted_repos)} repositories"
        )
        print(f"Top 10 repositories by example count:")
        for i, (repo_name, examples) in enumerate(sorted_repos[:10]):
            print(f"  {i+1}. {repo_name}: {len(examples)} examples")

        # Process each repository
        all_results = {}
        successful_repos = 0
        failed_repos = 0
        skipped_repos = 0
        skipped_repo_list = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        progress_file = os.path.join(
            script_dir, CONFIG["output"]["results_tracking_file"]
        )

        # Store original working directory
        original_cwd = os.getcwd()

        for repo_key, examples in tqdm(sorted_repos, desc="Processing repositories"):
            # if repo_key in repo_names_processed:
            #     continue
            # if not repo_key == "nccgroup-libslub-7732a54":
            #     continue
            # try:
            # Change back to original directory before processing each repo
            os.chdir(original_cwd)

            results, status = process_repository(
                repo_key,
                examples,
                neo4j_config,
                graph,
                model,
                tokenizer,
                retriever_type,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if status == "success":
                all_results[repo_key] = results
                successful_repos += 1
                save_results_tracking(
                    repo_name=repo_key,
                    exact_match_rate=results["summary"]["em_rate"],
                    average_edit_similarity=results["summary"][
                        "average_edit_similarity"
                    ],
                    total_examples=results["summary"]["total_examples"],
                    filename=progress_file,
                )
            elif status == "skipped":
                skipped_repos += 1
                skipped_repo_list.append(repo_key)
            else:  # failed
                failed_repos += 1

        # except KeyboardInterrupt:
        #     print("\n\nProcessing interrupted by user")
        #     break
        # except Exception as e:
        #     print(f"✗ Unexpected error with {repo_key}: {e}")
        #     failed_repos += 1
        #     continue

        # finally:
        #     # Restore original working directory
        #     os.chdir(original_cwd)

        # Summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total repositories: {len(sorted_repos)}")
        print(f"Successfully processed: {successful_repos}")
        print(f"Skipped (not found/private): {skipped_repos}")
        print(f"Failed (other errors): {failed_repos}")

        # Show some skipped repositories
        if skipped_repo_list:
            print(f"\nSkipped repositories (first 10):")
            for repo in skipped_repo_list[:10]:
                print(f"  - {repo}")
            if len(skipped_repo_list) > 10:
                print(f"  ... and {len(skipped_repo_list) - 10} more")

        # Save consolidated results
        if all_results:
            print("\nSaving consolidated results...")

            # Create summary of all repositories - simplified to only EM and ES
            summary_data = []
            for repo_name, results in all_results.items():
                summary_data.append(
                    {
                        "repo_name": repo_name,
                        "total_examples": results["summary"]["total_examples"],
                        "exact_match_rate": results["summary"]["em_rate"],
                        "average_edit_similarity": results["summary"][
                            "average_edit_similarity"
                        ],
                    }
                )

            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(CONFIG["output"]["summary_file"], index=False)

            # Save detailed results
            with open(CONFIG["output"]["detailed_results_file"], "w") as f:
                json.dump(all_results, f, indent=2)

            # Save skipped repositories list
            with open(CONFIG["output"]["skipped_repos_file"], "w") as f:
                for repo in skipped_repo_list:
                    f.write(f"{repo}\n")

            print(f"✓ Consolidated results saved to {CONFIG['output']['summary_file']}")
            print(
                f"✓ Detailed results saved to {CONFIG['output']['detailed_results_file']}"
            )
            print(
                f"✓ Skipped repositories saved to {CONFIG['output']['skipped_repos_file']}"
            )

            # Print top performing repositories
            if not summary_df.empty:
                summary_df_sorted = summary_df.sort_values(
                    "exact_match_rate", ascending=False
                )
                print(f"\nTop 10 repositories by Exact Match rate:")
                for i, row in summary_df_sorted.head(10).iterrows():
                    print(
                        f"  {row['repo_name']}: EM: {row['exact_match_rate']:.3f}, ES: {row['average_edit_similarity']:.3f}"
                    )
        else:
            print("\nNo repositories were successfully processed.")

            # Still save the skipped repositories list
            if skipped_repo_list:
                with open(CONFIG["output"]["skipped_repos_file"], "w") as f:
                    for repo in skipped_repo_list:
                        f.write(f"{repo}\n")
                print(
                    f"✓ Skipped repositories saved to {CONFIG['output']['skipped_repos_file']}"
                )

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
