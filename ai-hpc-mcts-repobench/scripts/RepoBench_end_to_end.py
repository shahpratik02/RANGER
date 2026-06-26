#!/usr/bin/env python3
"""
RepoBench Repository Processing Script
Processes each repository in RepoBench dataset by:
1. Cloning the repository at the correct commit
2. Creating a graph representation
3. Evaluating retrieval performance
"""

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
import torch.nn.functional as F
import argparse
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5EncoderModel

# Import functions from the other files
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path
from src.utils.config import CONFIG
from src.eval.RepoBenchEvaluation import evaluate_retrieval, save_results
from src.core.semantic_retriever import build_index_from_graph, CHROMA_CLIENT, COLLECTION_NAME


def clone_repo_at_commit(
    repo_name, created_at, base_dir=None
):
    """
    Clone a repository and checkout to the latest commit before created_at

    Args:
        repo_name: Repository name (e.g., "owner/repo")
        created_at: ISO timestamp string
        base_dir: Base directory to store cloned repos

    Returns:
        tuple: (repo_path, commit_id, error_message) where repo_path is None if failed
    """
    if base_dir is None:
        base_dir = CONFIG['paths']['repobench_base']
    try:
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Clean repo name for directory (replace / with _)
        clean_repo_name = repo_name.replace("/", "_")
        repo_path = os.path.join(base_dir, clean_repo_name)

        # Remove existing directory if it exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Clone the repository
        print(f"  Cloning {repo_name}...")
        clone_url = f"https://github.com/{repo_name}.git"

        env = os.environ.copy()
        env.update({
            'GIT_TERMINAL_PROMPT': '0',  # Disable interactive prompts
            'GIT_ASKPASS': 'echo',       # Provide empty password
            'SSH_ASKPASS': 'echo',       # Provide empty SSH password
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
            return None, None, f"Repository {repo_name} clone timed out (likely private or network issue)"

        # Check if clone was successful
        if result.returncode != 0:
            error_msg = result.stderr.strip()

            # Check for specific error types
            if (
                "repository not found" in error_msg.lower()
                or "not found" in error_msg.lower()
            ):
                return None, None, f"Repository {repo_name} not found (404)"
            elif (
                "permission denied" in error_msg.lower()
                or "forbidden" in error_msg.lower()
            ):
                return None, None, f"Repository {repo_name} is private or access denied (403)"
            elif "could not resolve host" in error_msg.lower():
                return None, None, f"Network error while cloning {repo_name}"
            else:
                return None, None, f"Git clone failed for {repo_name}: {error_msg}"

        # Change to repo directory
        os.chdir(repo_path)

        # Parse the created_at timestamp
        # try:
        #     created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        # except ValueError:
        #     # Handle different timestamp formats
        #     try:
        #         created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S")
        #     except ValueError:
        #         print(
        #             f"  Warning: Could not parse created_at '{created_at}', using current time"
        #         )
        #         created_date = datetime.now()

        created_date = datetime(2023, 12, 31, 23, 59, 59)

        # Get the latest commit before created_at
        print(f"  Finding latest commit before {created_date}...")
        result = subprocess.run(
            [
                "git",
                "rev-list",
                "-n",
                "1",
                f"--before={created_date.isoformat()}",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  Warning: Could not find commits before {created_at}, using HEAD")
            target_commit = "HEAD"
        else:
            target_commit = result.stdout.strip()

            if not target_commit:
                print(f"  Warning: No commits found before {created_at}, using HEAD")
                target_commit = "HEAD"

        # Checkout to the target commit
        print(
            f"  Checking out to commit: {target_commit[:8] if target_commit != 'HEAD' else 'HEAD'}..."
        )
        result = subprocess.run(
            ["git", "checkout", target_commit], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"  Warning: Checkout failed, staying on current branch")
            # Get current commit if checkout failed
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            )
            if result.returncode == 0:
                target_commit = result.stdout.strip()

        print(f"  ✓ Successfully prepared repository at {repo_path}")
        return repo_path, target_commit, None

    except Exception as e:
        return None, None, f"Unexpected error cloning repository {repo_name}: {e}"


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
    dummy_requirements = os.path.join(repo_path, "dummy_requirements.txt")
    with open(dummy_requirements, "w") as f:
        f.write("# Dummy requirements file\n")

    return dummy_requirements


def save_accuracy_tracking(
    repo_name,
    accuracy,
    mrr,
    ndcg,
    total_examples,
    correct_retrievals,
    filename="repobench_accuracy_tracking_graph_qwen3_8b_2.json",
):
    """
    Save or update the accuracy tracking JSON file after each repository

    Args:
        repo_name: Name of the repository
        accuracy: Accuracy score
        mrr: Mean Reciprocal Rank score
        ndcg: Normalized Discounted Cumulative Gain score
        total_examples: Total number of examples
        correct_retrievals: Number of correct retrievals
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
        "accuracy": accuracy,
        "mrr": mrr,
        "ndcg": ndcg,
        "total_examples": total_examples,
        "correct_retrievals": correct_retrievals,
        "processed_at": datetime.now().isoformat(),
    }

    # Save updated data
    with open(filename, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"✓ Updated accuracy tracking: {repo_name} - Accuracy: {accuracy:.3f}, MRR: {mrr:.3f}, nDCG: {ndcg:.3f}")


def create_graph_subprocess(repo_path, requirements_path, neo4j_config, graph):
    """Run create_graph as a subprocess for better performance"""
    """Run generate_graph.py as a subprocess with arguments"""
    try:
        # Clear database before creating graph
        clear_neo4j_database(graph)
        
        # Use configuration
        venv_python = CONFIG['paths']['venv_python']
        script_path = CONFIG['paths']['generate_graph_script']
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
            cmd, cwd=repo_path, capture_output=True, text=True, timeout=CONFIG['processing']['graph_generation_timeout']
        )

        if result.returncode == 0:
            return True, None
        else:
            return False, f"Graph creation failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"Graph creation timed out after {CONFIG['processing']['graph_generation_timeout']} seconds"
    except Exception as e:
        return False, f"Error running graph creation: {e}"


def save_neo4j_dump(repo_name, commit_id, dump_directory=None):
    """
    Save the current Neo4j database as a dump file and upload to Hugging Face.
    Uses commit ID instead of timestamp for better traceability.
    
    Args:
        repo_name: Repository name to use in dump filename
        commit_id: Git commit ID to use in dump filename
        dump_directory: Directory to store dump files
    
    Returns:
        bool: True if dump was created and uploaded successfully, False otherwise
    """
    if dump_directory is None:
        dump_directory = CONFIG['paths']['neo4j_dumps']
    import time
    try:
        # Create dump directory if it doesn't exist
        os.makedirs(dump_directory, exist_ok=True)
        
        # Create safe filename from repo name and commit ID
        safe_repo_name = repo_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        # Use first 8 characters of commit ID (standard short commit format)
        short_commit = commit_id[:8] if commit_id and commit_id != "HEAD" else "HEAD"
        dump_filename = f"{safe_repo_name}_{short_commit}.dump"
        dump_path = os.path.join(dump_directory, dump_filename)
        
        print(f"  📦 Creating dump for {repo_name} (commit: {short_commit})...")
        
        # Step 1: Stop Neo4j
        print("  ⏹️  Stopping Neo4j...")
        stop_cmd = ["sudo", "systemctl", "stop", "neo4j"]
        stop_result = subprocess.run(stop_cmd, capture_output=True, text=True, timeout=30)
        
        if stop_result.returncode != 0:
            print(f"  ✗ Failed to stop Neo4j: {stop_result.stderr}")
            return False
            
        # Wait a moment for Neo4j to fully stop
        time.sleep(3)
        subprocess.run([
    "sudo", "chown", "neo4j:neo4j", 
    f"{dump_directory}/neo4j.dump"
], check=False)
        # Step 2: Create the dump
        print("  💾 Creating database dump...")
        dump_cmd = [
            "sudo", "-u", "neo4j", 
            "neo4j-admin", "database", "dump", "neo4j",
            "--to-path", dump_directory,
            "--overwrite-destination"
                    ]
        
        dump_result = subprocess.run(dump_cmd, capture_output=True, text=True, timeout=300)
        print(f"Dump command result: {dump_result}")
        subprocess.run([
    "sudo", "chown", "neo4j:neo4j", 
    f"{dump_directory}/neo4j.dump"
], check=False)
        subprocess.run([
    "sudo", "chown", "ec2-user:ec2-user", 
    f"{dump_directory}/neo4j.dump"
], check=False)
        # Step 3: Start Neo4j back up
        print("  ▶️  Starting Neo4j...")
        start_cmd = ["sudo", "systemctl", "start", "neo4j"]
        start_result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=30)
        
        if start_result.returncode != 0:
            print(f"  ⚠️  Warning: Failed to restart Neo4j: {start_result.stderr}")
            print("  🔧 You may need to manually restart Neo4j")
        else:
            # Wait for Neo4j to be ready
            print("  ⏳ Waiting for Neo4j to be ready...")
            time.sleep(10)
        
        # Check if dump was successful
        if dump_result.returncode == 0:
            print("Renaming dump")
            # Rename the default dump file to our custom name
            default_dump = os.path.join(dump_directory, "neo4j.dump")
            if os.path.exists(default_dump):
                # os.rename(default_dump, dump_path)
                rename_result = subprocess.run(
                    ["sudo", "mv", default_dump, dump_path], 
                    capture_output=True, text=True, timeout=30
                )
                if rename_result.returncode == 0:
                    # Change ownership to ec2-user for easier access
                    subprocess.run(
                        ["sudo", "chown", "ec2-user:ec2-user", dump_path], 
                        capture_output=True, text=True, timeout=30
                    )
                    print(f"  ✓ Neo4j dump saved: {dump_filename}")
                else:
                    print(f"  ✗ Failed to rename dump file: {rename_result.stderr}")
                    return False
                # Step 4: Upload to Hugging Face
                print(f"  🚀 Uploading {dump_filename} to Hugging Face...")
                upload_cmd = [
                    "huggingface-cli", "upload",
                    "Nutanix/RepoBench-neo4j",
                    dump_path,
                    "--repo-type", "dataset",
                    "--commit-message", f"Add Neo4j dump for {repo_name} (commit: {short_commit})"
                ]
                
                upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=600)
                
                if upload_result.returncode == 0:
                    print(f"  ✅ Successfully uploaded {dump_filename} to Hugging Face")
                    return True
                else:
                    print(f"  ❌ Failed to upload to Hugging Face: {upload_result.stderr}")
                    print(f"  💾 Local dump file kept at: {dump_path}")
                    return False
                    
            else:
                print(f"  ⚠️  Dump command succeeded but file not found: {default_dump}")
                return False
        else:
            print(f"  ✗ Failed to create Neo4j dump: {dump_result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Neo4j dump operation timed out")
        # Try to restart Neo4j if it was stopped
        try:
            subprocess.run(["sudo", "systemctl", "start", "neo4j"], timeout=30)
        except:
            pass
        return False
    except Exception as e:
        print(f"  ✗ Error creating Neo4j dump: {e}")
        # Try to restart Neo4j if it was stopped
        try:
            subprocess.run(["sudo", "systemctl", "start", "neo4j"], timeout=30)
        except:
            pass
        return False


def clear_neo4j_database(graph, repo_name=None, commit_id=None):
    """Clear Neo4j database after saving dump if repo_name provided"""
    try:
        # Save dump before clearing if repo_name is provided

            
        graph.query("MATCH (n) DETACH DELETE (n)")
        print("  ✓ Database cleared")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not clear database: {e}")


def process_repository(repo_name, examples, neo4j_config, graph, model, tokenizer, retriever_type):
    """
    Process a single repository: clone, create graph, evaluate

    Args:
        repo_name: Repository name
        examples: List of examples for this repository
        graph: Neo4jGraph instance

    Returns:
        tuple: (results, status) where status is 'success', 'skipped', or 'failed'
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"Number of examples: {len(examples)}")
    print(f"{'='*60}")
    
    original_cwd = os.getcwd()
    try:
        # Get created_at from the first example
        first_example = examples[0]
        created_at = first_example.get("created_at", "2023-01-01T00:00:00Z")

        print(f"Repository created at: {created_at}")

        # Clone repository at the correct commit
        repo_path, commit_id, error_msg = clone_repo_at_commit(repo_name, created_at)
        if not repo_path:
            print(f"Skipping {repo_name}: {error_msg}")
            # INSERT_YOUR_CODE
            # Open repobench_skipped_repos.txt and add repo name
            try:
                with open("repobench_skipped_repos.txt", "a") as f:
                    f.write(f"{repo_name}\n")
            except Exception as e:
                print(f"Warning: Could not write to repobench_skipped_repos.txt: {e}")
            return None, "skipped"

        print(f"Using commit: {commit_id[:8] if commit_id and commit_id != 'HEAD' else 'HEAD'}")
        
        # Clear database with commit-aware dump creation
        print("Clearing Neo4j database...")
        clear_neo4j_database(graph, repo_name, commit_id)

        # Find requirements file
        requirements_path = find_requirements_file(repo_path)
        requirements_path=""
        print(f"Using requirements file: {requirements_path}")

        # Create graph representation
        print(f"Creating graph for {repo_name}...")
        try:
            # create_graph(
            #     root_dir=repo_path,
            #     requirements_path=requirements_path,
            #     url=neo4j_config["url"],
            #     username=neo4j_config["username"],
            #     password=neo4j_config["password"],
            # )
            success, error = create_graph_subprocess(
                repo_path, requirements_path, neo4j_config, graph
            )
            if success:
                save_neo4j_dump(repo_name, commit_id)
                print(f" Graph created successfully")

            else:
                print(f" Graph creation failed: {error}")
                return None, "failed"
        except Exception as e:
            print(f" Graph creation failed: {e}")
            return None, "failed"

        # If using semantic retriever, build the index now
        if retriever_type == "semantic":
            print("Building semantic index...")
            try:
                # Clear collection before building new index to avoid mixing repos
                try:
                    CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
                except Exception:
                    pass  # Collection might not exist, which is fine
                collection = CHROMA_CLIENT.get_or_create_collection(
                    name=COLLECTION_NAME
                )

                success = build_index_from_graph(repo_name, collection, neo4j_config)
                if success:
                    print("  ✓ Semantic index built successfully.")
                else:
                    print("  ✗ Semantic index building failed.")
                    return None, "failed"
            except Exception as e:
                print(f"  ✗ Error building semantic index: {e}")
                return None, "failed"

        # Evaluate retrieval performance
        print(f"Evaluating retrieval for {repo_name}...")

        # Query format from RepoBenchEvaluation.py
        query_format="""
Given file_name: {file_name}
Fetch the most important depnedencies from the repo to complete the following code, FOCUS ON THE BOTTOM INCOMPLETE CODE ONLY:
{code}"""

        # Create dataset from examples for evaluation
        repo_dataset = examples

        try:
            results = evaluate_retrieval(
                dataset=repo_dataset,
                query_format=query_format,
                repo_filter=repo_name,
                retriever_type=retriever_type,
                repo_name_for_semantic=repo_name,
                model=model,
                tokenizer=tokenizer,
                neo4j_config=neo4j_config,
                graph=graph
            )

            # Print summary
            print("\n" + "=" * 40)
            print("EVALUATION SUMMARY")
            print("=" * 40)
            print(f"Total Examples: {results['summary']['total_examples']}")
            print(f"Correct Retrievals: {results['summary']['correct_retrievals']}")
            print(f"Accuracy: {results['summary']['accuracy']:.3f}")
            print(f"MRR: {results['summary']['mrr']:.3f}")
            print(f"nDCG: {results['summary']['ndcg']:.3f}")
            print(f"Average Similarity: {results['summary']['average_similarity']:.3f}")
            print(f"Max Similarity: {results['summary']['max_similarity']:.3f}")
            print(f"Min Similarity: {results['summary']['min_similarity']:.3f}")

            # Save results
            safe_repo_name = repo_name.replace("/", "_")
            results_path = os.path.join(original_cwd, f"repobench_eval_{safe_repo_name}_results.csv")

            save_results(results, f"{original_cwd}/results_graph_unixcoder/repobench_eval_{safe_repo_name}")

            return results, "success"

        except Exception as e:
            print(f" Evaluation failed: {e}")
            return None, "failed"

    except Exception as e:
        print(f"Error processing repository {repo_name}: {e}")
        return None, "failed"


def main():
    """Main function to process all RepoBench repositories"""
    parser = argparse.ArgumentParser(description="RepoBench Processing Script")
    parser.add_argument(
        "--retriever",
        type=str,
        default="graph",
        choices=["graph", "semantic", "file_level"],
        help="Retriever to use for evaluation ('graph', 'semantic', or 'file_level')",
    )
    args = parser.parse_args()
    print(f"Using retriever: {args.retriever}")

    # Use configuration
    neo4j_config = CONFIG['database']['neo4j']
    
    url = neo4j_config['url']
    username = neo4j_config['username']
    password = neo4j_config['password']
    graph = Neo4jGraph(url=url, username=username, password=password)
    import json

    # Read the JSON file
    progress_filename = CONFIG['paths']['progress_file']
    with open(progress_filename, 'r') as file:
        acc_data = json.load(file)

    # Get all repository names (keys)
    processed_repo_names = list(acc_data.keys())

    # Print all repository names
    

    # Or if you want them as a list
    try:
        # Use configuration for model loading
        model_config = CONFIG['models']['embedding']
        EMBEDDING_MODEL = model_config['name']
        print(f"Loading {EMBEDDING_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, 
            padding_side=model_config['padding_side']
        )
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            trust_remote_code=model_config['trust_remote_code'], 
            torch_dtype=getattr(torch, model_config['torch_dtype'])
        )
           
        # Move to GPU if available
        device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
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

        # Sort repositories by number of examples (descending)
        sorted_groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_groups = [item for item in sorted_groups if len(item[1]) >= 5]
        print(f"Dataset contains {len(dataset_list)} examples from {len(grouped)} repositories")
        print(f"After filtering (>5 examples): {len(sorted_groups)} repositories")
        print(f"Top 10 repositories by example count:")
        for i, (repo_name, examples) in enumerate(sorted_groups[:10]):
            print(f"  {i+1}. {repo_name}: {len(examples)} examples")

        # Process each repository
        all_results = {}
        successful_repos = 0
        failed_repos = 0
        skipped_repos = 0
        skipped_repo_list = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        progress_file = os.path.join(script_dir, CONFIG['paths']['progress_file'])
        # Store original working directory
        original_cwd = os.getcwd()

        try:
            for repo_name, examples in tqdm(sorted_groups, desc="Processing repositories"):
                if repo_name  in processed_repo_names:
                    continue
                # if  not repo_name=='DLYuanGod/TinyGPT-V':
                #     continue
                try:
                    # Change back to original directory before processing each repo
                    os.chdir(original_cwd)

                    results, status = process_repository(repo_name, examples, neo4j_config, graph, model, tokenizer, args.retriever)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if status == "success":
                        all_results[repo_name] = results
                        successful_repos += 1
                        save_accuracy_tracking(
                            repo_name=repo_name,
                            accuracy=results["summary"]["accuracy"],
                            mrr=results["summary"]["mrr"],
                            ndcg=results["summary"]["ndcg"],
                            total_examples=results["summary"]["total_examples"],
                            correct_retrievals=results["summary"]["correct_retrievals"],
                            filename=progress_file,
                        )
                    elif status == "skipped":
                        skipped_repos += 1
                        skipped_repo_list.append(repo_name)
                    else:  # failed
                        failed_repos += 1

                except KeyboardInterrupt:
                    print("\n\nProcessing interrupted by user")
                    break
                except Exception as e:
                    print(f"✗ Unexpected error with {repo_name}: {e}")
                    failed_repos += 1
                    continue
                

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

        # Summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total repositories: {len(sorted_groups)}")
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

            # Create summary of all repositories
            summary_data = []
            for repo_name, results in all_results.items():
                summary_data.append(
                    {
                        "repo_name": repo_name,
                        "total_examples": results["summary"]["total_examples"],
                        "correct_retrievals": results["summary"]["correct_retrievals"],
                        "accuracy": results["summary"]["accuracy"],
                        "mrr": results["summary"]["mrr"],
                        "ndcg": results["summary"]["ndcg"],
                        "average_similarity": results["summary"]["average_similarity"],
                        "max_similarity": results["summary"]["max_similarity"],
                        "min_similarity": results["summary"]["min_similarity"],
                    }
                )

            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(CONFIG['paths']['summary_csv'], index=False)

            # Save detailed results
            with open(CONFIG['paths']['detailed_json'], "w") as f:
                json.dump(all_results, f, indent=2)

            # Save skipped repositories list
            with open("repobench_skipped_repos.txt", "w") as f:
                for repo in skipped_repo_list:
                    f.write(f"{repo}\n")

            print("✓ Consolidated results saved to repobench_all_repos_*")
            print("✓ Skipped repositories saved to repobench_skipped_repos.txt")

            # Print top performing repositories
            if not summary_df.empty:
                summary_df_sorted = summary_df.sort_values("accuracy", ascending=False)
                print(f"\nTop 10 repositories by accuracy:")
                for i, row in summary_df_sorted.head(10).iterrows():
                    print(f"  {row['repo_name']}: Accuracy: {row['accuracy']:.3f}, MRR: {row['mrr']:.3f}, nDCG: {row['ndcg']:.3f}")

        else:
            print("\nNo repositories were successfully processed.")

            # Still save the skipped repositories list
            if skipped_repo_list:
                with open("repobench_skipped_repos.txt", "w") as f:
                    for repo in skipped_repo_list:
                        f.write(f"{repo}\n")
                print("✓ Skipped repositories saved to repobench_skipped_repos.txt")

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
