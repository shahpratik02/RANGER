#!/usr/bin/env python3
"""
RepoBench Evaluation Script
Evaluates a retrieval system against the RepoBench dataset
"""

import json
import math
from collections import defaultdict
from difflib import SequenceMatcher
from datasets import load_dataset
from src.core.retirever_v2 import main, stage_1, stage_cross_module_deps
from src.core.semantic_retriever import main as semantic_main
import pandas as pd
from tqdm import tqdm
from src.utils.config import CONFIG


def is_25_percent_similar(text1, text2):
    """Check if two texts are 95% similar using SequenceMatcher"""
    matcher = SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    return similarity >= 0.25, similarity


def is_75_percent_similar(text1, text2):
    """Check if two texts are 95% similar using SequenceMatcher"""
    matcher = SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    return similarity >= 0.75, similarity


def evaluate_retrieval(
    dataset,
    query_format,
    max_examples=None,
    repo_filter=None,
    graph=None,
    model=None,
    tokenizer=None,
    retriever_type="graph",
    repo_name_for_semantic=None,
    neo4j_config=None,
):
    """
    Evaluate retrieval system on RepoBench dataset

    Args:
        dataset: HuggingFace dataset
        query_format: Format string for queries
        max_examples: Maximum number of examples to process (None for all)
        repo_filter: Specific repository to filter by (None for all)
        retriever_type: Type of retriever to use ('graph' or 'semantic')
        repo_name_for_semantic: The name of the repo, required for semantic
    Returns:
        dict: Evaluation results
    """

    results = []
    total_examples = 0
    correct_retrievals = 0
    similarity_scores = []
    reciprocal_ranks = []
    dcg_scores = []

    # Filter examples if repo_filter is specified
    if repo_filter:
        examples = [item for item in dataset if item["repo_name"] == repo_filter]
    else:
        examples = list(dataset)

    # Limit examples if specified
    if max_examples:
        examples = examples[:max_examples]

    print(f"Processing {len(examples)} examples...")

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        print("example " ,i)
        try:
            # Format the query
            path = example["file_path"]
            if path.startswith("tme/tests/"):
                path = path.replace(
                    "tme/tests/", "tests/", 1
                )  # Replace only the first occurrence
            # Convert file path to module path
            if path.endswith("/__init__.py"):
                path = path[:-12]  # Remove /__init__.py
            elif path.endswith(".py"):
                path = path[:-3]  # Remove .py
            
            path = path.replace('/', '.')
            question = query_format.format(file_name=path, code=example["cropped_code"])

            # Get retrieval results
            if retriever_type == "semantic":
                if not repo_name_for_semantic:
                    raise ValueError(
                        "repo_name_for_semantic must be provided for semantic retriever"
                    )
                retrieved_results = semantic_main(
                    question=question,
                    repo_name=repo_name_for_semantic,
                    neo4j_config=neo4j_config,
                )
            elif retriever_type == "file_level":
                # Convert file path to module name for file_level retriever
                module_name = path.replace("/", ".").replace(".py", "")
                if module_name.endswith(".__init__"):
                    module_name = module_name[:-9]  # Remove .__init__
                
                retrieved_results = stage_cross_module_deps(
                    graph=graph, 
                    module_name=module_name, 
                    question=question, 
                    model=model, 
                    tokenizer=tokenizer
                )
            else:  # Default to graph retriever
                retrieved_results = stage_1(graph, question, model, tokenizer)


            # Get ground truth
            gold_snippet = example["context"][example["gold_snippet_index"]]["snippet"]
            gold_snippet_name = example["context"][example["gold_snippet_index"]][
                "identifier"
            ]

            # Check if any retrieved result matches the gold snippet
            found_match = False
            best_similarity = 0.0
            best_match_idx = -1
            match_rank = 0
            if not isinstance(retrieved_results, list):
                retrieved_results = []
            for j, result in enumerate(retrieved_results):
                if result["code"] is None:
                    continue
                is_similar, similarity = is_25_percent_similar(
                    gold_snippet, result["code"]
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = j
                if "name" in result:
                    if (
                        is_similar
                        and gold_snippet_name.strip() == result["name"].strip()
                    ):
                        found_match = True
                        match_rank = j + 1
                        print("example", i, "Success with Name")
                        break
                else:
                    is_similar, similarity = is_75_percent_similar(
                        gold_snippet, result["code"]
                    )
                    if is_similar:
                        found_match = True
                        match_rank = j + 1
                        print("example", i, "Success")
                        break

            # Store detailed results
            result_entry = {
                "example_idx": i,
                "repo_name": example["repo_name"],
                "file_path": example["file_path"],
                "found_match": found_match,
                "best_similarity": best_similarity,
                "best_match_idx": best_match_idx,
                "num_retrieved": len(retrieved_results),
                "retrieved_signatures": [r["signature"] for r in retrieved_results],
            }
            if found_match:
                result_entry["match_rank"] = match_rank

            results.append(result_entry)

            # Update counters
            total_examples += 1
            if found_match:
                correct_retrievals += 1
                reciprocal_ranks.append(1 / match_rank)
                dcg_scores.append(1 / math.log2(match_rank + 1))
            else:
                reciprocal_ranks.append(0)
                dcg_scores.append(0)


            similarity_scores.append(best_similarity)

            # Print progress every 10 examples
            if (i + 1) % 10 == 0:
                current_accuracy = correct_retrievals / total_examples
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                current_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
                current_ndcg = sum(dcg_scores) / len(dcg_scores)
                print(
                    f"Progress: {i+1}/{len(examples)} | "
                    f"Accuracy: {current_accuracy:.3f} | "
                    f"Avg Similarity: {avg_similarity:.3f} | "
                    f"MRR: {current_mrr:.3f} | "
                    f"nDCG: {current_ndcg:.3f}"
                )

        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            # Store error information
            error_entry = {
                "example_idx": i,
                "repo_name": example["repo_name"],
                "file_path": example["file_path"],
                "error": str(e),
                "found_match": False,
                "best_similarity": 0.0,
            }
            results.append(error_entry)
            total_examples += 1
            reciprocal_ranks.append(0)
            dcg_scores.append(0)

    # Calculate final metrics
    accuracy = correct_retrievals / total_examples if total_examples > 0 else 0
    avg_similarity = (
        sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    )
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    ndcg = sum(dcg_scores) / len(dcg_scores) if dcg_scores else 0

    evaluation_summary = {
        "total_examples": total_examples,
        "correct_retrievals": correct_retrievals,
        "accuracy": accuracy,
        "mrr": mrr,
        "ndcg": ndcg,
        "average_similarity": avg_similarity,
        "max_similarity": max(similarity_scores) if similarity_scores else 0,
        "min_similarity": min(similarity_scores) if similarity_scores else 0,
    }

    return {"summary": evaluation_summary, "detailed_results": results}


def analyze_by_repository(results):
    """Analyze results grouped by repository"""
    repo_stats = {}

    for result in results["detailed_results"]:
        if "error" not in result:
            repo = result["repo_name"]
            if repo not in repo_stats:
                repo_stats[repo] = {"total": 0, "correct": 0, "similarities": [], "reciprocal_ranks": [], "dcg_scores": []}
            
            repo_stats[repo]["total"] += 1
            if result["found_match"]:
                repo_stats[repo]["correct"] += 1
                match_rank = result["match_rank"]
                repo_stats[repo]["reciprocal_ranks"].append(1 / match_rank)
                repo_stats[repo]["dcg_scores"].append(1 / math.log2(match_rank + 1))
            else:
                repo_stats[repo]["reciprocal_ranks"].append(0)
                repo_stats[repo]["dcg_scores"].append(0)
            repo_stats[repo]["similarities"].append(result["best_similarity"])

    # Calculate per-repo metrics
    repo_analysis = {}
    for repo, stats in repo_stats.items():
        repo_analysis[repo] = {
            "total_examples": stats["total"],
            "correct_retrievals": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "mrr": sum(stats["reciprocal_ranks"]) / len(stats["reciprocal_ranks"]) if stats["reciprocal_ranks"] else 0,
            "ndcg": sum(stats["dcg_scores"]) / len(stats["dcg_scores"]) if stats["dcg_scores"] else 0,
            "avg_similarity": (
                sum(stats["similarities"]) / len(stats["similarities"])
                if stats["similarities"]
                else 0
            ),
        }

    return repo_analysis


def save_results(results, filename_prefix="repobench_eval"):
    """Save results to files"""

    # Save summary
    # with open(f"{filename_prefix}_summary.json", "w") as f:
    #     json.dump(results["summary"], f, indent=2)

    # # Save detailed results
    # with open(f"{filename_prefix}_detailed.json", "w") as f:
    #     json.dump(results["detailed_results"], f, indent=2)

    # Create DataFrame for easier analysis
    df = pd.DataFrame(results["detailed_results"])
    df.to_csv(f"{filename_prefix}_results.csv", index=False)

    print(
        f"Results saved to {filename_prefix}_results.csv"
    )


def main_evaluation():
    """Main evaluation function"""

    # Load dataset
    print("Loading RepoBench dataset...")
    dataset = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first")

    # Group by repository to see distribution
    grouped = {}
    dataset_list = list(dataset)
    for item in dataset_list:
        repo_name = item["repo_name"]
        grouped[repo_name] = grouped.get(repo_name, 0) + 1

    sorted_groups = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    print(f"Dataset contains {len(dataset_list)} examples from {len(grouped)} repositories")

    # Query format
    query_format = """
Given file_name: {file_name}
Fetch the most important connected nodes from the graph to complete the below code. FOCUS ON THE BOTTOM INCOMPLETE CODE ONLY
{code}"""

    # Run evaluation on the most frequent repository first (for testing)
    top_repo = sorted_groups[7][0]
    print(f"\nRunning evaluation on top repository: {top_repo}")

    results = evaluate_retrieval(
        dataset=dataset_list,
        query_format=query_format,
        repo_filter=top_repo,
    )

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Examples: {results['summary']['total_examples']}")
    print(f"Correct Retrievals: {results['summary']['correct_retrievals']}")
    print(f"Accuracy: {results['summary']['accuracy']:.3f}")
    print(f"MRR: {results['summary']['mrr']:.3f}")
    print(f"nDCG: {results['summary']['ndcg']:.3f}")
    print(f"Average Similarity: {results['summary']['average_similarity']:.3f}")
    print(f"Max Similarity: {results['summary']['max_similarity']:.3f}")
    print(f"Min Similarity: {results['summary']['min_similarity']:.3f}")

    # Save results
    save_results(results, f"repobench_eval_{top_repo.replace('/', '_')}")

    # For full evaluation, uncomment below:
    # print("\nRunning full evaluation...")
    # full_results = evaluate_retrieval(
    #     dataset=dataset_list,
    #     query_format=query_format,
    #     max_examples=None  # Process all examples
    # )
    #
    # repo_analysis = analyze_by_repository(full_results)
    # save_results(full_results, "repobench_eval_full")
    #
    # with open("repobench_repo_analysis.json", 'w') as f:
    #     json.dump(repo_analysis, f, indent=2)


if __name__ == "__main__":
    main_evaluation()
