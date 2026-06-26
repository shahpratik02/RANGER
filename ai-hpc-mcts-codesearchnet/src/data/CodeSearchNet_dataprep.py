import json
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from langchain_neo4j import Neo4jGraph
import re
from src.utils.simple_config import CONFIG


@dataclass
class CodeSearchNetDoc:
    """Represents a code document from the corpus"""

    doc_id: str
    repo: str
    path: str
    func_name: str
    code: str
    language: str


class GraphCorpusFilter:
    """Filter knowledge graph to keep only corpus nodes and their dependencies"""

    def __init__(self, url: str, username: str, password: str):
        """Initialize the Neo4j graph connection."""
        self.graph = Neo4jGraph(url=url, username=username, password=password)
        self.nodes_to_keep = set()

    def path_to_module_name(self, path: str) -> str:
        """
        Convert file path to module name.
        Example: 'dlkit/json_/osid/default_mdata.py' -> 'dlkit.json_.osid.default_mdata'
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

    def parse_func_name(self, func_name: str) -> Tuple[Optional[str], str]:
        """
        Parse function name to extract class and method if it's in ClassName.method_name format.

        Returns:
            (class_name, method_name) if it's a method
            (None, func_name) if it's a regular function
        """
        if "." in func_name:
            # This is a method in format ClassName.method_name
            parts = func_name.split(".")
            if len(parts) == 2:
                class_name, method_name = parts
                return class_name, method_name
            else:
                # Handle cases like Module.Class.method - take last two parts
                class_name = parts[-2]
                method_name = parts[-1]
                return class_name, method_name
        else:
            # This is a regular function
            return None, func_name

    def extract_corpus_targets(
        self, corpus: List[CodeSearchNetDoc]
    ) -> List[Dict[str, Any]]:
        """
        Extract target nodes from corpus.
        Returns list of dicts with module_name, func_name, class_name (if method).
        """
        targets = []

        for doc in corpus:
            module_name = self.path_to_module_name(doc.path)
            class_name, method_name = self.parse_func_name(doc.func_name)

            target = {
                "module_name": module_name,
                "func_name": method_name,
                "original_func_name": doc.func_name,
                "path": doc.path,
                "is_method": class_name is not None,
                "class_name": class_name,
            }

            targets.append(target)

        return targets

    def find_corpus_nodes(self, targets: List[Dict[str, Any]]) -> Set[str]:
        """
        Find all nodes in the graph that correspond to corpus functions.
        Returns set of node IDs.
        """
        corpus_nodes = set()

        for target in targets:
            module_name = target["module_name"]
            func_name = target["func_name"]
            is_method = target["is_method"]
            class_name = target["class_name"]
            original_func_name = target["original_func_name"]

            print(f"Looking for '{original_func_name}' in module '{module_name}'")

            if is_method:
                # Look for a Method node
                query = """
                MATCH (m:Method {name: $func_name, class: $class_name, module_name: $module_name})
                RETURN elementId(m) as node_id, m.name as name, m.class as class, m.module_name as module, 'Method' as type
                """

                result = self.graph.query(
                    query,
                    {
                        "func_name": func_name,
                        "class_name": class_name,
                        "module_name": module_name,
                    },
                )

                if result:
                    for node in result:
                        node_id = node["node_id"]
                        corpus_nodes.add(node_id)
                        print(
                            f"  ✓ Found Method: {node['class']}.{node['name']} in {node['module']}"
                        )
                else:
                    print(
                        f"  ✗ Method '{class_name}.{func_name}' not found in module '{module_name}'"
                    )

                    # Also try without exact module match in case of module name mismatch
                    fallback_query = """
                    MATCH (m:Method {name: $func_name, class: $class_name})
                    RETURN elementId(m) as node_id, m.name as name, m.class as class, m.module_name as module, 'Method' as type
                    """

                    fallback_result = self.graph.query(
                        fallback_query,
                        {"func_name": func_name, "class_name": class_name},
                    )

                    if fallback_result:
                        for node in fallback_result:
                            node_id = node["node_id"]
                            corpus_nodes.add(node_id)
                            print(
                                f"  ✓ Found Method (fallback): {node['class']}.{node['name']} in {node['module']}"
                            )
                    else:
                        print(
                            f"  ✗ Method '{class_name}.{func_name}' not found anywhere"
                        )
            else:
                # Look for a Function node
                query = """
                MATCH (f:Function {name: $func_name, module_name: $module_name})
                RETURN elementId(f) as node_id, f.name as name, f.module_name as module, 'Function' as type
                """

                result = self.graph.query(
                    query, {"func_name": func_name, "module_name": module_name}
                )

                if result:
                    for node in result:
                        node_id = node["node_id"]
                        corpus_nodes.add(node_id)
                        print(f"  ✓ Found Function: {node['name']} in {node['module']}")
                else:
                    print(
                        f"  ✗ Function '{func_name}' not found in module '{module_name}'"
                    )

                    # Also try without exact module match in case of module name mismatch
                    fallback_query = """
                    MATCH (f:Function {name: $func_name})
                    RETURN elementId(f) as node_id, f.name as name, f.module_name as module, 'Function' as type
                    """

                    fallback_result = self.graph.query(
                        fallback_query, {"func_name": func_name}
                    )

                    if fallback_result:
                        for node in fallback_result:
                            node_id = node["node_id"]
                            corpus_nodes.add(node_id)
                            print(
                                f"  ✓ Found Function (fallback): {node['name']} in {node['module']}"
                            )
                    else:
                        print(f"  ✗ Function '{func_name}' not found anywhere")

        return corpus_nodes

    def find_dependencies(self, node_ids: Set[str], corpus_nodes: Set[str]) -> Set[str]:
        """
        Find all nodes that the given nodes depend on (via USES relationships).
        Returns set of node IDs including the original nodes.
        """
        all_dependencies = set(node_ids)
        to_explore = set(node_ids)

        while to_explore:
            current_batch = to_explore.copy()
            to_explore.clear()

            # Convert set to list for Cypher query
            batch_list = list(current_batch)

            # Find all nodes that current batch USES
            query = """
            UNWIND $node_ids as node_id
            MATCH (n)-[:USES]->(target)
            WHERE elementId(n) = node_id
            RETURN DISTINCT elementId(target) as target_id
            """

            result = self.graph.query(query, {"node_ids": batch_list})

            for record in result:
                target_id = record["target_id"]
                if target_id not in all_dependencies and target_id in corpus_nodes:
                    all_dependencies.add(target_id)
                    to_explore.add(target_id)

        print(f"Found {len(all_dependencies)} nodes including dependencies")
        return all_dependencies

    def find_path_to_root(self, node_ids: Set[str]) -> Set[str]:
        """
        Find all nodes on the path from given nodes to the root (Repo).
        This includes all CONTAINS relationships going upward.
        """
        all_path_nodes = set(node_ids)
        to_explore = set(node_ids)

        while to_explore:
            current_batch = to_explore.copy()
            to_explore.clear()

            # Convert set to list for Cypher query
            batch_list = list(current_batch)

            # Find all nodes that CONTAIN current batch
            query = """
            UNWIND $node_ids as node_id
            MATCH (container)-[:CONTAINS]->(n)
            WHERE elementId(n) = node_id
            RETURN DISTINCT elementId(container) as container_id
            """

            result = self.graph.query(query, {"node_ids": batch_list})

            for record in result:
                container_id = record["container_id"]
                if container_id not in all_path_nodes:
                    all_path_nodes.add(container_id)
                    to_explore.add(container_id)

        print(f"Found {len(all_path_nodes)} nodes including path to root")
        return all_path_nodes

    def find_class_and_method_relationships(self, node_ids: Set[str]) -> Set[str]:
        """
        Find all classes that contain methods in our node set,
        and all methods that belong to classes in our node set.
        Also find inheritance relationships.
        """
        additional_nodes = set()
        batch_list = list(node_ids)

        # Find classes that have methods in our set
        query1 = """
        UNWIND $node_ids as node_id
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE elementId(m) = node_id
        RETURN DISTINCT elementId(c) as class_id
        """

        result1 = self.graph.query(query1, {"node_ids": batch_list})
        for record in result1:
            additional_nodes.add(record["class_id"])

        # Find methods that belong to classes in our set
        query2 = """
        UNWIND $node_ids as node_id
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE elementId(c) = node_id
        RETURN DISTINCT elementId(m) as method_id
        """

        result2 = self.graph.query(query2, {"node_ids": batch_list})
        for record in result2:
            additional_nodes.add(record["method_id"])

        # Find fields that belong to classes in our set
        query3 = """
        UNWIND $node_ids as node_id
        MATCH (c:Class)-[:HAS_FIELD]->(f:Field)
        WHERE elementId(c) = node_id
        RETURN DISTINCT elementId(f) as field_id
        """

        result3 = self.graph.query(query3, {"node_ids": batch_list})
        for record in result3:
            additional_nodes.add(record["field_id"])

        # Find parent classes (via INHERITS relationships)
        query4 = """
        UNWIND $node_ids as node_id
        MATCH (c:Class)-[:INHERITS]->(parent:Class)
        WHERE elementId(c) = node_id
        RETURN DISTINCT elementId(parent) as parent_id
        """

        result4 = self.graph.query(query4, {"node_ids": batch_list})
        for record in result4:
            additional_nodes.add(record["parent_id"])

        # Find child classes (via INHERITS relationships)
        query5 = """
        UNWIND $node_ids as node_id
        MATCH (child:Class)-[:INHERITS]->(c:Class)
        WHERE elementId(c) = node_id
        RETURN DISTINCT elementId(child) as child_id
        """

        result5 = self.graph.query(query5, {"node_ids": batch_list})
        for record in result5:
            additional_nodes.add(record["child_id"])

        print(
            f"Found {len(additional_nodes)} additional class/method/field/inheritance nodes"
        )
        return additional_nodes

    def filter_graph(self, corpus: List[CodeSearchNetDoc], repo_name: str) -> Dict[str, int]:
        """
        Main method to filter the graph based on corpus.
        Returns statistics about the filtering process.
        """
        print("🔍 Starting graph filtering process...")

        # Step 1: Extract targets from corpus
        targets = self.extract_corpus_targets(corpus)
        print(f"📝 Processing {len(targets)} corpus functions")

        # Print summary of what we're looking for
        functions_count = sum(1 for t in targets if not t["is_method"])
        methods_count = sum(1 for t in targets if t["is_method"])
        print(f"  - {functions_count} functions")
        print(f"  - {methods_count} methods")

        # Step 2: Find corpus nodes in graph
        corpus_nodes = self.find_corpus_nodes(targets)
        if not corpus_nodes:
            print("❌ No corpus nodes found in graph!")
            return {"error":0}

        # Step 3: Find all dependencies
        print("🔗 Finding dependencies...")
        nodes_with_deps = self.find_dependencies(corpus_nodes, corpus_nodes)

        # Step 4: Find path to root
        print("🌲 Finding path to root...")
        nodes_with_path = self.find_path_to_root(nodes_with_deps)

        # Step 5: Find related class/method relationships
        print("🏗️ Finding class/method relationships...")
        class_method_nodes = self.find_class_and_method_relationships(nodes_with_path)

        # Step 6: IMPORTANT FIX - Find path to root for newly added classes
        print("🌲 Finding path to root for additional class/method nodes...")
        additional_path_nodes = self.find_path_to_root(class_method_nodes)

        # Step 7: Combine all nodes to keep
        all_nodes_to_keep = nodes_with_path | class_method_nodes | additional_path_nodes

        # Step 8: Get statistics before deletion
        stats = self.get_graph_statistics()
        print(f"📊 Graph statistics before filtering:")
        for node_type, count in stats.items():
            print(f"  {node_type}: {count}")

        # Step 9: Delete nodes not in our keep set
        print("🗑️ Deleting unnecessary nodes...")
        deleted_count = self.delete_nodes_not_in_set(all_nodes_to_keep, repo_name)
        # deleted_count = 0
        # Step 10: Get final statistics
        final_stats = self.get_graph_statistics()
        print(f"📊 Graph statistics after filtering:")
        for node_type, count in final_stats.items():
            print(f"  {node_type}: {count}")

        return {
            "corpus_functions": len(targets),
            "corpus_nodes_found": len(corpus_nodes),
            "total_nodes_kept": len(all_nodes_to_keep),
            "nodes_deleted": deleted_count,
            "functions_count": functions_count,
            "methods_count": methods_count,
            "original_stats": stats,
            "final_stats": final_stats,
        }

    def get_graph_statistics(self) -> Dict[str, int]:
        """Get count of each node type in the graph."""
        query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY node_type
        """

        result = self.graph.query(query)
        stats = {}
        for record in result:
            node_type = record["node_type"]
            count = record["count"]
            stats[node_type] = count

        return stats

    def delete_nodes_not_in_set(self, nodes_to_keep: Set[str], repo_name: str) -> int:
        """
        Delete all nodes that are not in the nodes_to_keep set.
        Returns count of deleted nodes.
        """
        if repo_name:
            # Get all node IDs belonging to this repository
            repo_nodes_query = """
            MATCH (r:Repo {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(m:Module)
            OPTIONAL MATCH (m)-[:CONTAINS]->(child)
            OPTIONAL MATCH (child)-[:HAS_METHOD]->(method)
            WITH r, m, child, method
            RETURN elementId(r) as node_id
            UNION
            MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)
            RETURN elementId(m) as node_id
            UNION
            MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(child)
            RETURN elementId(child) as node_id
            UNION
            MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(child)-[:HAS_METHOD]->(method)
            RETURN elementId(method) as node_id
            """
            
            all_nodes = self.graph.query(repo_nodes_query, {"repo_name": repo_name})
            all_node_ids = {record["node_id"] for record in all_nodes}
            
            print(f"Found {len(all_node_ids)} nodes in repository {repo_name}")
        else:
            return 0

        # Find nodes to delete
        nodes_to_delete = all_node_ids - nodes_to_keep

        if not nodes_to_delete:
            print("No nodes to delete")
            return 0

        print(f"Deleting {len(nodes_to_delete)} nodes...")

        # Delete nodes in batches to avoid memory issues
        batch_size = 1000
        deleted_count = 0

        nodes_to_delete_list = list(nodes_to_delete)

        for i in range(0, len(nodes_to_delete_list), batch_size):
            batch = nodes_to_delete_list[i : i + batch_size]

            delete_query = """
            UNWIND $node_ids as node_id
            MATCH (n)
            WHERE elementId(n) = node_id
            DETACH DELETE n
            RETURN count(n) as deleted_in_batch
            """

            result = self.graph.query(delete_query, {"node_ids": batch})
            batch_deleted = result[0]["deleted_in_batch"] if result else 0
            deleted_count += batch_deleted

            if i % (batch_size * 10) == 0:  # Print progress every 10 batches
                print(f"  Deleted {i + len(batch)} / {len(nodes_to_delete_list)} nodes")

        print(f"✅ Successfully deleted {deleted_count} nodes")
        return deleted_count

    def close(self):
        """Close the graph connection."""
        self.graph.close()


# def load_corpus_from_list(data: List) -> List[CodeSearchNetDoc]:
#     """
#     Load corpus from JSON file.
#     Expected format: list of dictionaries with CodeSearchNetDoc fields.
#     """
#     # with open(file_path, "r") as f:
#     #     data = json.load(f)

#     corpus = []
#     for item in data:
#         corpus.append(
#             CodeSearchNetDoc(
#                 doc_id=item.doc_id,
#                 repo=item.repo,
#                 path=item.path,
#                 func_name=item.func_name,
#                 code=item.code,
#                 language=item.language,
#             )
#         )

#     return corpus


def main(corpus, repo_name):
    """Main function to run the filtering process."""
    # Configuration
    neo4j_config = CONFIG['neo4j']
    url = neo4j_config['url']
    username = neo4j_config['username']
    password = neo4j_config['password']

    # Example corpus - replace with your actual corpus

    # Alternative: Load from JSON file
    # corpus = load_corpus_from_json('corpus.json')

    # Initialize filter
    # corpus = load_corpus_from_list(data)
    filter_tool = GraphCorpusFilter(url, username, password)

    try:
        # Run the filtering process
        results = filter_tool.filter_graph(corpus, repo_name)

        print("\n🎉 Filtering completed!")
        print(f"Results: {json.dumps(results, indent=2)}")

    except Exception as e:
        print(f"❌ Error during filtering: {str(e)}")
        raise
    finally:
        filter_tool.close()


if __name__ == "__main__":
    main()
