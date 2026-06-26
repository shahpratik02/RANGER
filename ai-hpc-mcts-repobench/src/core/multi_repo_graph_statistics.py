#!/usr/bin/env python3
"""
Multi-Repository Graph Statistics Script
Analyzes a Neo4j database containing multiple repositories and generates statistics for each.
"""

import os
import json
import sys
from datetime import datetime
from langchain_neo4j import Neo4jGraph

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from src.utils.config import CONFIG


def get_all_repositories(graph):
    """Get all repository names from the database"""
    query = """
    MATCH (r:Repo)
    RETURN r.name as repo_name
    ORDER BY r.name
    """
    result = graph.query(query)
    return [record["repo_name"] for record in result]


def get_repository_statistics(graph, repo_name):
    """Get statistics for a specific repository"""
    
    # Query to get node counts by type for a specific repository
    node_count_query = """
    MATCH (r:Repo {name: $repo_name})
    OPTIONAL MATCH (r)-[:CONTAINS*]->(n)
    WHERE n:Module OR n:Class OR n:Function OR n:Method OR n:GlobalVariable OR n:Field
    RETURN labels(n)[0] as node_type, count(n) as count
    ORDER BY node_type
    """
    
    # Query to get total nodes connected to this repository
    total_nodes_query = """
    MATCH (r:Repo {name: $repo_name})
    OPTIONAL MATCH (r)-[:CONTAINS*]->(n)
    RETURN count(n) as total_nodes
    """
    
    # Query to get Python files count (assuming Module nodes represent Python files)
    python_files_query = """
    MATCH (r:Repo {name: $repo_name})
    OPTIONAL MATCH (r)-[:CONTAINS]->(m:Module)
    RETURN count(m) as python_files
    """
    
    try:
        # Get node counts by type
        result = graph.query(node_count_query, {"repo_name": repo_name})
        
        # Initialize statistics
        stats = {
            "repo_name": repo_name,
            "total_nodes": 0,
            "node_counts": {
                "Module": 0,
                "Class": 0, 
                "Function": 0,
                "Method": 0,
                "GlobalVariable": 0,
                "Field": 0
            },
            "python_files": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process node counts
        for record in result:
            node_type = record["node_type"]
            count = record["count"]
            if node_type and node_type in stats["node_counts"]:
                stats["node_counts"][node_type] = count
        
        # Get total nodes
        total_result = graph.query(total_nodes_query, {"repo_name": repo_name})
        if total_result:
            stats["total_nodes"] = total_result[0]["total_nodes"] or 0
        
        # Get Python files count
        files_result = graph.query(python_files_query, {"repo_name": repo_name})
        if files_result:
            stats["python_files"] = files_result[0]["python_files"] or 0
        
        return stats
        
    except Exception as e:
        print(f"Error getting statistics for {repo_name}: {e}")
        return None


def save_statistics(all_stats, filename="csn_graph_statistics.json"):
    """Save all repository statistics to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"✓ Statistics saved to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving statistics: {e}")
        return False


def print_summary(all_stats):
    """Print a summary of the statistics"""
    if not all_stats:
        print("No statistics to display")
        return
    
    print(f"\n{'='*80}")
    print("MULTI-REPOSITORY GRAPH STATISTICS SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories analyzed: {len(all_stats)}")
    
    # Calculate totals
    total_nodes_all = sum(stats["total_nodes"] for stats in all_stats.values())
    total_python_files_all = sum(stats["python_files"] for stats in all_stats.values())
    
    print(f"Total nodes across all repositories: {total_nodes_all:,}")
    print(f"Total Python files across all repositories: {total_python_files_all:,}")
    
    # Node type totals
    node_type_totals = {
        "Module": 0,
        "Class": 0,
        "Function": 0,
        "Method": 0,
        "GlobalVariable": 0,
        "Field": 0
    }
    
    for stats in all_stats.values():
        for node_type, count in stats["node_counts"].items():
            if node_type in node_type_totals:
                node_type_totals[node_type] += count
    
    print(f"\nNode type breakdown across all repositories:")
    for node_type, total in node_type_totals.items():
        print(f"  {node_type}: {total:,}")
    
    # Top repositories by total nodes
    sorted_repos = sorted(all_stats.items(), key=lambda x: x[1]["total_nodes"], reverse=True)
    print(f"\nTop 10 repositories by total nodes:")
    for i, (repo_name, stats) in enumerate(sorted_repos[:10]):
        print(f"  {i+1:2d}. {repo_name}: {stats['total_nodes']:,} nodes, {stats['python_files']} Python files")
    
    # Repositories with no nodes (potential issues)
    empty_repos = [repo_name for repo_name, stats in all_stats.items() if stats["total_nodes"] == 0]
    if empty_repos:
        print(f"\n⚠️  Repositories with no nodes ({len(empty_repos)}):")
        for repo_name in empty_repos[:5]:  # Show first 5
            print(f"  - {repo_name}")
        if len(empty_repos) > 5:
            print(f"  ... and {len(empty_repos) - 5} more")


def main():
    """Main function to analyze multi-repository graph statistics"""
    # Use configuration for Neo4j connection
    # neo4j_config = CONFIG['database']['neo4j']
    neo4j_config = {'url': "bolt://localhost:7687",
    'username': "neo4j" ,
    'password': "your_neo4j_password"}  # Update with your Neo4j password
    try:
        # Connect to Neo4j
        print("Connecting to Neo4j database...")
        graph = Neo4jGraph(
            url=neo4j_config['url'], 
            username=neo4j_config['username'], 
            password=neo4j_config['password']
        )
        
        # Get all repositories
        print("Getting list of repositories...")
        all_repo_names = get_all_repositories(graph)
        
        if not all_repo_names:
            print("No repositories found in the database")
            return
        
        print(f"Found {len(all_repo_names)} repositories in the database")
        
        # Analyze each repository
        all_stats = {}
        successful = 0
        failed = 0
        
        print(f"\nAnalyzing repositories...")
        for i, repo_name in enumerate(all_repo_names, 1):
            print(f"  [{i:3d}/{len(all_repo_names):3d}] {repo_name}...", end=" ")
            
            stats = get_repository_statistics(graph, repo_name)
            if stats:
                all_stats[repo_name] = stats
                successful += 1
                print(f"✓ ({stats['total_nodes']} nodes, {stats['python_files']} files)")
            else:
                failed += 1
                print("✗ Failed")
        
        # Save results
        if all_stats:
            save_statistics(all_stats)
            print_summary(all_stats)
        else:
            print("No statistics were successfully collected")
        
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully analyzed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output saved to: csn_graph_statistics.json")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        # Close graph connection
        try:
            graph.close()
            print("✓ Graph connection closed")
        except Exception as e:
            print(f"Warning: Error closing graph connection: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
