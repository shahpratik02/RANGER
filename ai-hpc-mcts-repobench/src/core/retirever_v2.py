import openai
import httpx
import json
import concurrent.futures
import os
import json
import fnmatch
import argparse
from pathlib import Path
from argparse import ArgumentParser
from time import time
import logging
import httpx
from tqdm import tqdm
import yaml
import gc
from transformers import AutoTokenizer
import transformers
import openai
from typing import Any, Dict, List, Optional, Union, Tuple
import requests
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from transformers import AutoModel
from typing import List, Dict, Any
import re
from src.utils.prompts import (
    entity_extractor_prompt,
    repobench_query_generation_prompt_concise,
)
from src.utils.schema import nodes_description, edges_description
from src.utils.config import CONFIG



def last_token_pool(last_hidden_states, attention_mask):
    """Extract embeddings using last token pooling (for Qwen3)"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_llm_response(query: Dict) -> str:
    """
    This function uses the OpenAI API to generate text based on a given prompt and model from the aws inference endpoint.
    The function returns a string containing the generated text.

    Args:
        user_prompt (str): The prompt used to generate the text. Defaults to an empty string.
        system_prompt (str): The system prompt. Defaults to an empty string.
        url (str): The URL of the OpenAI API.
        model (str): The model used to generate the text.
        stream (bool): Whether to stream the generation process. Defaults to True.
        kwargs (dict): Additional keyword arguments for the OpenAI API. Defaults to an empty dictionary.

    Returns:
        str: The generated text from the OpenAI API.
    """

    api_key = "EMPTY"
    base_url = query["url"]
    model = query["model"]
    client = openai.OpenAI(
        api_key=api_key, base_url=base_url, http_client=httpx.Client(verify=False)
    )
    # api_key = getattr(args, "API_KEY", "EMPTY")
    # client = openai.OpenAI(api_key=api_key, base_url=query["url"])
    # client = openai.OpenAI(api_key=args.API_KEY, base_url=query['url'])
    # print(query['system_prompt'])
    # print(query['user_prompt'])
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": query["system_prompt"],
                },
                {
                    "role": "user",
                    "content": query["user_prompt"],
                },
            ],
            stream=True,
            **query.get("kwargs", {}),
        )
    except Exception:
        print("Error in OpenAI API call")
    gen_tokens = ""
    output = list(stream)
    for idx, i in enumerate(output):
        if idx > 0:
            content = i.choices[0].delta.content
            if content:
                gen_tokens += i.choices[0].delta.content

    return gen_tokens.strip()


def extract_json(text):
    # Find the first {...} JSON object using brace counting
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
            if count == 0:
                json_str = text[start : i + 1]
                try:
                    return json.loads(json_str)
                except Exception as e:
                    print("JSON decode error:", e)
                    return None
    return None


class CypherQueryGenerator:
    def __init__(self, graph):
        """
        Initialize with Neo4j graph connection

        Args:
            graph: Neo4jGraph instance
        """
        self.graph = graph
        # self.schema_label_mapping={
        #     "MODULE":"Module",
        #     "CLASS":"Class",
        #     "METHOD":"Method",
        #     "FUNCTION":"Function",
        #     "GLOBAL_VARIABLE":"GlobalVariable",
        #     "FIELD":"Field",
        #     }

    def generate_cypher_from_json(self, extracted_json: Dict[str, Any]) -> List[str]:
        """
        Generate Cypher queries from the extracted JSON object

        Args:
            extracted_json: JSON object from LLM entity extraction

        Returns:
            List of Cypher query strings
        """
        queries = []

        # Handle entities - can be single dict or list of dicts
        if extracted_json is None:
            return []
        entities = extracted_json.get("entities", {})
        if isinstance(entities, dict) and entities:
            # Single entity dict
            query = self._build_entity_query(entities)
            if query:
                queries.append(query)
        elif isinstance(entities, list):
            # List of entity dicts
            for entity in entities:
                if entity:
                    query = self._build_entity_query(entity)
                    if query:
                        queries.append(query)

        # Handle relationships - can be single dict or list of dicts
        relationships = extracted_json.get("relationships", {})
        if isinstance(relationships, dict) and relationships:
            # Single relationship dict
            query = self._build_generic_relationship_query(relationships)
            if query:
                queries.append(query)
        elif isinstance(relationships, list):
            # List of relationship dicts
            for relationship in relationships:
                if relationship:
                    query = self._build_generic_relationship_query(relationship)
                    if query:
                        queries.append(query)

        return queries

    def _build_entity_query(self, entities: Dict[str, str]) -> Optional[str]:
        """Build query for finding specific entities"""
        entity_type = entities.get("type")
        entity_name = entities.get("name")

        if not entity_name:
            return None

        # Case-insensitive name matching
        if entity_type:
            if entity_type == "Module":
                query = f"""
                    MATCH (n:{entity_type})
                    WHERE toLower(n.name) CONTAINS toLower('{entity_name}')
                    RETURN n, labels(n) as node_type
                    LIMIT 5
                    """

            else:
                query = f"""
                MATCH (n:{entity_type})
                WHERE toLower(n.name) = toLower('{entity_name}')
                RETURN n, labels(n) as node_type
                LIMIT 5
                """
        else:
            # Search across all node types when type is ambiguous
            query = f"""
            MATCH (n)
            WHERE toLower(n.name) = toLower('{entity_name}')
            RETURN n, labels(n) as node_type
            LIMIT 5
            """

        return query.strip()

    def _build_generic_relationship_query(
        self, relationships: Dict[str, str]
    ) -> Optional[str]:
        """Build query for relationships without specific entity"""
        rel_type = relationships.get("type")
        rel_source = relationships.get("source")
        rel_target = relationships.get("target")

        if not rel_type:
            return None

        conditions = []
        if rel_source:
            conditions.append(
                f"toLower(source.name) CONTAINS|USES  toLower('{rel_source}')"
            )
        if rel_target:
            conditions.append(
                f"toLower(target.name) CONTAINS|USES toLower('{rel_target}')"
            )

        where_clause = " AND ".join(conditions)
        where_clause = f"WHERE {where_clause}" if where_clause else ""

        query = f"""
        MATCH (source)-[r:{rel_type}]->(target)
        {where_clause}
        RETURN source, r, target, labels(source) as source_type, labels(target) as target_type
        LIMIT 5
        """

        return query.strip()

    def execute_queries(self, extracted_json: Dict[str, Any]) -> List:
        """
        Generate and execute Cypher queries, return unique nodes

        Args:
            extracted_json: JSON object from LLM entity extraction

        Returns:
            List of unique nodes
        """
        queries = self.generate_cypher_from_json(extracted_json)
        print("Stage 2 queries created \n ", queries)
        unique_nodes_keys = set()
        unique_nodes = []

        for query in queries:

            try:
                result = self.graph.query(query)
                # print(query)
                # Extract all nodes from the result

                for record in result:
                    if "node_type" in record:
                        key = ()
                        # Use a tuple of node properties as the key for uniqueness
                        if record["node_type"][0] == "Module":
                            key = ("Module", record["n"]["name"])
                        elif record["node_type"][0] == "Class":
                            key = (
                                "Class",
                                record["n"].get("name"),
                                record["n"].get("module_name"),
                                record["n"].get("signature"),
                            )
                        elif record["node_type"][0] == "Function":
                            key = (
                                "Function",
                                record["n"].get("name"),
                                record["n"].get("module_name"),
                                record["n"].get("signature"),
                            )
                        elif record["node_type"][0] == "Method":
                            key = (
                                "Method",
                                record["n"].get("name"),
                                record["n"].get("class"),
                                record["n"].get("signature"),
                            )
                        elif record["node_type"][0] == "Field":
                            key = (
                                "Field",
                                record["n"].get("name"),
                                record["n"].get("class"),
                            )
                        elif record["node_type"][0] == "GlobalVariable":
                            key = (
                                "GlobalVariable",
                                record["n"].get("name"),
                                record["n"].get("module_name"),
                            )

                        if key not in unique_nodes_keys:
                            unique_nodes_keys.add(key)
                            record["n"]["node_type"] = record["node_type"][
                                0
                            ]  # Add node type for clarity
                            unique_nodes.append(record["n"])
                        # Add the node to the set of unique nodes
                    elif "source" in record and "target" in record and "r" in record:
                        # Process source node
                        source_node = record["source"]
                        source_type = (
                            record["source_type"][0]
                            if record["source_type"]
                            else "Unknown"
                        )

                        # Create unique key for source node
                        source_key = ()
                        if source_type == "Module":
                            source_key = ("Module", source_node.get("name"))
                        elif source_type == "Class":
                            source_key = (
                                "Class",
                                source_node.get("name"),
                                source_node.get("module_name"),
                                source_node.get("signature"),
                            )
                        elif source_type == "Function":
                            source_key = (
                                "Function",
                                source_node.get("name"),
                                source_node.get("module_name"),
                                source_node.get("signature"),
                            )
                        elif source_type == "Method":
                            source_key = (
                                "Method",
                                source_node.get("name"),
                                source_node.get("class"),
                                source_node.get("signature"),
                            )
                        elif source_type == "Field":
                            source_key = (
                                "Field",
                                source_node.get("name"),
                                source_node.get("class"),
                            )
                        elif source_type == "GlobalVariable":
                            source_key = (
                                "GlobalVariable",
                                source_node.get("name"),
                                source_node.get("module_name"),
                            )

                        # Add source node if not already present
                        if source_key not in unique_nodes_keys:
                            unique_nodes_keys.add(source_key)
                            source_node["node_type"] = source_type
                            unique_nodes.append(source_node)

                        # Process target node
                        target_node = record["target"]
                        target_type = (
                            record["target_type"][0]
                            if record["target_type"]
                            else "Unknown"
                        )

                        # Create unique key for target node
                        target_key = ()
                        if target_type == "Module":
                            target_key = ("Module", target_node.get("name"))
                        elif target_type == "Class":
                            target_key = (
                                "Class",
                                target_node.get("name"),
                                target_node.get("module_name"),
                                target_node.get("signature"),
                            )
                        elif target_type == "Function":
                            target_key = (
                                "Function",
                                target_node.get("name"),
                                target_node.get("module_name"),
                                target_node.get("signature"),
                            )
                        elif target_type == "Method":
                            target_key = (
                                "Method",
                                target_node.get("name"),
                                target_node.get("class"),
                                target_node.get("signature"),
                            )
                        elif target_type == "Field":
                            target_key = (
                                "Field",
                                target_node.get("name"),
                                target_node.get("class"),
                            )
                        elif target_type == "GlobalVariable":
                            target_key = (
                                "GlobalVariable",
                                target_node.get("name"),
                                target_node.get("module_name"),
                            )

                        # Add target node if not already present
                        if target_key not in unique_nodes_keys:
                            unique_nodes_keys.add(target_key)
                            target_node["node_type"] = target_type
                            unique_nodes.append(target_node)

                        # # Process relationship
                        # relationship = record["r"]
                        # rel_key = (
                        #     source_key,
                        #     relationship.type,  # Relationship type
                        #     target_key
                        # )

                        # # Add relationship if not already present
                        # if rel_key not in unique_relationships_keys:
                        #     unique_relationships_keys.add(rel_key)
                        #     relationship_data = {
                        #         "source": source_node,
                        #         "target": target_node,
                        #         "relationship": relationship,
                        #         "relationship_type": relationship.type
                        #     }
                        #     unique_relationships.append(relationship_data)

            except Exception as e:
                print(f"Query error: {e}")
                continue
        return unique_nodes

    def _fetch_connected_nodes_with_embeddings(self, node) -> List:
        """
        Fetch nodes connected to the given node via USES relationship that have embeddings

        Args:
            node: The source node

        Returns:
            List of connected nodes with embeddings
        """
        # Get node identifier based on its type
        node_labels = node["node_type"]
        if not node_labels:
            return []

        node_type = node_labels

        # Build query to find connected nodes via USES relationship
        if node_type == "Module":
            identifier = f"n.name = '{node['name']}'"
        elif node_type == "Class":
            identifier = f"n.name = '{node['name']}' AND n.module_name = '{node.get('module_name', '')}'"
        elif node_type == "Function":
            identifier = f"n.name = '{node['name']}' AND n.module_name = '{node.get('module_name', '')}'"
        elif node_type == "Method":
            identifier = (
                f"n.name = '{node['name']}' AND n.class = '{node.get('class', '')}'"
            )
        elif node_type == "Field":
            identifier = (
                f"n.name = '{node['name']}' AND n.class = '{node.get('class', '')}'"
            )
        elif node_type == "GlobalVariable":
            identifier = f"n.name = '{node['name']}' AND n.module_name = '{node.get('module_name', '')}'"
        else:
            return []
        if node_type == "Module":
            query = f"""
            MATCH (n:{node_type})-[:CONTAINS]->(connected)
            WHERE {identifier} AND (connected.embedding IS NOT NULL)
            RETURN connected, labels(connected) as connected_type
            """
        else:
            query = f"""
            MATCH (n:{node_type})-[:USES]->(connected)
            WHERE {identifier} AND (connected.embedding IS NOT NULL)
            RETURN connected, labels(connected) as connected_type
            """

        try:
            result = self.graph.query(query)
            return [
                (record["connected"], record["connected_type"]) for record in result
            ]
        except Exception as e:
            print(f"Error fetching connected nodes: {e}")
            return []

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays and reshape for sklearn
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def execute_queries_with_ranking(
        self,
        extracted_json: Dict[str, Any],
        question_embedding: List[float],
        context_limit: int = 8000,
    ) -> List[str]:
        """
        Execute queries, fetch connected nodes, rank by similarity, and return code within context limit

        Args:
            extracted_json: JSON object from LLM entity extraction
            question_embedding: Embedding vector of the question
            context_limit: Maximum character limit for returned code

        Returns:
            List of code strings ordered by similarity score
        """
        # Get initial nodes from queries
        initial_nodes = self.execute_queries(extracted_json)
        # Collect all nodes with embeddings (initial + connected)
        all_nodes_with_embeddings = []

        # Add initial nodes that have embeddings
        for node in initial_nodes:
            if hasattr(node, "get") and node.get("embedding"):
                all_nodes_with_embeddings.append(node)
        # Fetch connected nodes for each initial node
        for node in initial_nodes:
            connected_nodes = self._fetch_connected_nodes_with_embeddings(node)
            i = 0
            for node, node_type in connected_nodes:
                if hasattr(node, "get") and node.get("embedding"):
                    # Add node type for clarity
                    node["node_type"] = (
                        node_type[0] if isinstance(node_type, list) else node_type
                    )
                    all_nodes_with_embeddings.append(node)
                    i += 1
            # print(f"Fetched {i} connected nodes")
        # Remove duplicates based on node properties
        unique_nodes_dict = {}
        for node in all_nodes_with_embeddings:
            node_labels = node.get("node_type", [])
            if not node_labels:
                continue

            node_type = node_labels

            # Create unique key for deduplication
            if node_type == "Module":
                key = ("Module", node.get("name"))
            elif node_type == "Class":
                key = (
                    "Class",
                    node.get("name"),
                    node.get("module_name"),
                    node.get("signature"),
                )
            elif node_type == "Function":
                key = (
                    "Function",
                    node.get("name"),
                    node.get("module_name"),
                    node.get("signature"),
                )
            elif node_type == "Method":
                key = (
                    "Method",
                    node.get("name"),
                    node.get("class"),
                    node.get("signature"),
                )
            elif node_type == "Field":
                key = ("Field", node.get("name"), node.get("class"))
            elif node_type == "GlobalVariable":
                key = ("GlobalVariable", node.get("name"), node.get("module_name"))
            else:
                continue

            unique_nodes_dict[key] = node

        # Calculate similarities and create ranking
        node_similarities = []
        for node in unique_nodes_dict.values():
            if node.get("embedding") and node.get("code"):
                similarity = self._calculate_similarity(
                    question_embedding, node["embedding"]
                )
                node_similarities.append((node, similarity))

        # Sort by similarity score (descending)
        node_similarities.sort(key=lambda x: x[1], reverse=True)

        # Collect code within context limit
        result_codes = []
        current_length = 0

        for node, similarity in node_similarities:
            code = node.get("code", "")
            if not code:
                continue

            # Check if adding this code would exceed context limit
            if current_length + len(code) > context_limit:
                break

            result_codes.append(code)
            current_length += len(code)

        return result_codes


def find_similar_nodes(graph, question_embedding, top_k=1):
    """
    Find top K nodes with highest cosine similarity to question embedding

    Args:
        graph: Neo4jGraph instance
        question_embedding: List or array of embedding values
        top_k: Number of top similar nodes to return (default: 5)

    Returns:
        List of dictionaries containing node info and similarity scores
    """

    # Convert embedding to proper Cypher list format
    if isinstance(question_embedding, np.ndarray):
        question_embedding = question_embedding.tolist()

    # Format as proper Cypher list with comma-separated values
    embedding_values = ", ".join([str(float(x)) for x in question_embedding])
    embedding_str = f"[{embedding_values}]"

    # Cypher query to find similar nodes using GDS cosine similarity
    cypher_query = f"""
    MATCH (n)
    WHERE n.embedding IS NOT NULL
    WITH n, gds.similarity.cosine(n.embedding, {embedding_str}) AS similarity
    RETURN n, similarity
    ORDER BY similarity DESC
    LIMIT {top_k}
    """

    try:
        result = graph.query(cypher_query)
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return []


def create_embedding(text, embedding_model=None):
    embedding = embedding_model.encode(text)
    return embedding


def extract_cypher_query(text):
    # Try to find a query inside ```cypher ... ```
    cypher_block = re.search(r"```cypher\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if cypher_block:
        return cypher_block.group(1).strip()
    # If not found, try to find the first MATCH ... RETURN ... pattern
    match = re.search(
        r"(MATCH\s.*?RETURN\s.*?)(?:\n|$)", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    # If not found, return the whole text (if it looks like a query)
    if "MATCH" in text and "RETURN" in text:
        return text.strip()
    return None


def split_code_recursively(
    code: str, prefix: str, tokenizer, max_doc_length: int
) -> List[str]:
    """
    Recursively split code into equal halves until all chunks fit within context limit
    """
    # Create full document
    full_doc = prefix + f" | Code: {code}" if prefix else f"Code: {code}"

    # Check if it fits
    if len(tokenizer.encode(full_doc)) <= max_doc_length:
        return [full_doc]

    # Split code in half
    lines = code.split('\n')
    if len(lines) > 1:
        mid_line = len(lines) // 2
        left_half = "\n".join(lines[:mid_line])
        right_half = "\n".join(lines[mid_line:])
    else:
        # If only one line, split by character as a fallback
        mid = len(code) // 2
        left_half = code[:mid]
        right_half = code[mid:]

    # Recursively split both halves
    left_chunks = split_code_recursively(left_half, prefix, tokenizer, max_doc_length)
    right_chunks = split_code_recursively(right_half, prefix, tokenizer, max_doc_length)

    return left_chunks + right_chunks



def rerank_results(
    question: str,
    results: List[Dict],
    model,
    tokenizer,
    batch_size: int = 4,  # Reduced batch size for embedding model
    max_length: int = 512,  # Reduced for Qwen3-8B context limit
    top_k: int = 5,
) -> List[Dict]:
    """
    Rerank results using embedding-based similarity with Qwen3-8B
    """
    if not results:
        return results

    device = next(model.parameters()).device
    
    # Calculate space for documents
    question_length = len(tokenizer.encode(question))
    max_doc_length = max_length - question_length - 10

    # Prepare document chunks
    all_chunks = []
    chunk_to_result_map = []

    for result_idx, result in enumerate(results):
        # Create prefix (name + signature)
        prefix_parts = []
        if result.get("name"):
            prefix_parts.append(f"Name: {result['name']}")
        if result.get("signature"):
            prefix_parts.append(f"Signature: {result['signature']}")

        prefix = " | ".join(prefix_parts)

        # Handle code
        if result.get("code"):
            chunks = split_code_recursively(
                result["code"], prefix, tokenizer, max_doc_length
            )
        else:
            chunks = [prefix] if prefix else [""]

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_result_map.append(result_idx)

    print(f"Processing {len(all_chunks)} chunks from {len(results)} results")

    # Generate query embedding once
    with torch.no_grad():
        query_inputs = tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        query_outputs = model(**query_inputs)
        query_embedding = last_token_pool(query_outputs.last_hidden_state, query_inputs['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)  # Normalize
        query_embedding = query_embedding.cpu().numpy()

    # Process document chunks in batches to get embeddings
    all_chunk_embeddings = []

    with torch.no_grad():
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            all_chunk_embeddings.extend(batch_embeddings)
            del inputs
            del outputs

    # Calculate cosine similarities
    similarities = []
    for chunk_embedding in all_chunk_embeddings:
        # Compute cosine similarity
        similarity = np.dot(query_embedding[0], chunk_embedding)
        similarities.append(float(similarity))

    # Take max similarity for each result
    result_scores = [float("-inf")] * len(results)
    for chunk_idx, similarity in enumerate(similarities):
        result_idx = chunk_to_result_map[chunk_idx]
        result_scores[result_idx] = max(result_scores[result_idx], similarity)

    # Sort by score
    scored_results = [
        {"result": result, "score": score}
        for result, score in zip(results, result_scores)
    ]
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    
    if top_k is not None:
        scored_results = scored_results[:top_k]
        print(f"Returning top {len(scored_results)} results")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return [item["result"] for item in scored_results]

def stage_cross_module_deps(graph: Neo4jGraph, module_name: str, question: str, model, tokenizer) -> Optional[List]:
    """
    Find cross-module dependencies for a given module and rerank them based on question relevance
    
    Args:
        graph: Neo4jGraph instance
        module_name: Name of the module to find dependencies for
        question: Question for reranking relevance
        model: Reranking model
        tokenizer: Tokenizer for the reranking model
    
    Returns:
        List of reranked target nodes or None if no results
    """
    
    cypher_query = """
    MATCH (sourceModule:Module {name: $module_name})
    MATCH (sourceModule)-[:CONTAINS]->(directChild)
    OPTIONAL MATCH (sourceModule)-[:CONTAINS]->(class:Class)-[:HAS_METHOD]->(method:Method)

    WITH sourceModule.name as source_module_name, collect(directChild) + collect(method) as allSourceNodes

    UNWIND allSourceNodes as sourceNode
    MATCH (sourceNode)-[:USES]->(target)
    WHERE target.module_name <> source_module_name

    RETURN DISTINCT
      target.name as name,
      target.code as code,
      target.signature as signature,
      target.module_name as module_name,
      labels(target)[0] as node_type
    ORDER BY target.module_name, target.name
    """
    
    print(f"Finding cross-module dependencies for module: {module_name}")
    
    try:
        result = graph.query(cypher_query, {"module_name": module_name})
        
        if not result:
            print("No cross-module dependencies found for the module.")
            return None
        else:
            print(f"Found {len(result)} cross-module dependencies. Reranking...")
            for res in result:
                print(f"  - {res['name']} (from {res['module_name']})")
            
            reranked_results = rerank_results(question, result, model, tokenizer, top_k=5)
            return reranked_results
            
    except Exception as e:
        print(f"Error executing cross-module dependency query: {e}")
        return None

def stage_1(graph: Neo4jGraph, question, model, tokenizer) -> Optional[str]:

    generator_prompt = repobench_query_generation_prompt_concise.format(
        nodes_description=nodes_description, edges_description=edges_description
    )

    query = {}
    query["system_prompt"] = generator_prompt
    query["user_prompt"] = question
    # query["url"] = "http://localhost:8000/v1"
    # query['url']="http://localhost:8000/v1"
    query['url']="http://localhost:8000/v1"
    query["model"] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    query["kwargs"] = {
        "temperature": 0.0,  # Lower value for less randomness
        "top_p": 0.99,  # Optional: controls nucleus sampling
        # "max_tokens": 512,  # Optional: limit response length
    }
    result = get_llm_response(query=query)
    print(result)
    try:
        cypher_query = extract_cypher_query(result)
        print("Stage 1 Cypher query: ", cypher_query)
    except Exception as e:
        print(f"Error extracting Cypher query: {e}")
        return None
    if not cypher_query:
        print("No valid Cypher query found in the response.")
        return None
    try:
        result = graph.query(cypher_query)

        if not result:
            print("No results found for the Cypher query.")
            return None
        else:
            print(f"Reranking {len(result)} results...")
            for res in result:
                print(res['name'])
            reranked_results = rerank_results(question, result, model, tokenizer, top_k=5)
            return reranked_results
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return None


# def stage_2(graph: Neo4jGraph, question_embedding: List[float], question) -> List[str]:
#     """
#     Stage 2: Execute queries, fetch connected nodes, rank by similarity, and return code within context limit

#     Args:
#         graph: Neo4jGraph instance
#         question_embedding: Embedding vector of the question

#     Returns:
#         List of code strings ordered by similarity score
#     """
#     # Create CypherQueryGenerator instance
#     QUESTION = "Tell me about the agent module and its capabilities."

#     extractor_prompt = entity_extractor_prompt.format(
#         nodes_description=nodes_description, edges_description=edges_description
#     )

#     query = {}
#     query["system_prompt"] = extractor_prompt
#     query["user_prompt"] = question
#     query["url"] = "http://3.234.239.168:8000/v1"
#     query["model"] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
#     query["kwargs"] = {
#         "temperature": 0.01,  # Lower value for less randomness
#         "top_p": 0.99,  # Optional: controls nucleus sampling
#         # "max_tokens": 512,  # Optional: limit response length
#     }

#     result = get_llm_response(query=query)
#     try:
#         extracted = extract_json(result)
#     except Exception as e:
#         print(f"Error extracting JSON: {e}")
#         return None

#     cypher_generator = CypherQueryGenerator(graph)

#     result = cypher_generator.execute_queries_with_ranking(
#         extracted_json=extracted, question_embedding=question_embedding
#     )
#     if not result:
#         print("No results found for the Cypher query.")
#         return None
#     else:
#         return result


# def stage_3(graph: Neo4jGraph, embedding_model, question) -> List[str]:
#     """
#     Stage 3: Find similar nodes based on question embedding

#     Args:
#         graph: Neo4jGraph instance
#         embedding_model: Embedding model
#         question: Question

#     Returns:
#         List of code strings ordered by similarity score
#     """
#     llm_config = {
#         "base_url": "http://localhost:11434/v1",
#         "model": "codellama:7b",
#         "api_key": "EMPTY",
#         "kwargs": {
#             "temperature": 0.01,  # Slightly higher for more nuanced scoring
#             "top_p": 0.95,  # Slightly lower for more focused responses
#         },
#     }

#     # Initialize components

#     # Initialize RL-Enhanced GraphRAG
#     rl_graphrag = RLEnhancedGraphRAG(
#         graph=graph,
#         embedding_model=embedding_model,
#         llm_config=llm_config,
#         max_iterations=10,
#         reward_threshold=6.0,
#         alpha=0.3,
#     )

#     # Example query
#     # query = "Tell me about the agent module and its capabilities."

#     # Perform search
#     high_reward_nodes = rl_graphrag.search(question)

#     # Display results
#     print(f"\nFound {len(high_reward_nodes)} high-reward nodes:")
#     for i, node_info in enumerate(high_reward_nodes, 1):
#         print(
#             f"{i}. {node_info['node_data'].get('name', 'Unknown')} "
#             f"(Reward: {node_info['avg_reward']:.2f}, "
#             f"Visits: {node_info['visit_count']})"
#         )

#     return [x["node_data"]["code"] for x in high_reward_nodes]


def main(question="Tell me about the agent module and its capabilities."):
    # Use configuration
    neo4j_config = CONFIG['database']['neo4j']
    
    url = neo4j_config['url']
    username = neo4j_config['username'] 
    password = neo4j_config['password']
    graph = Neo4jGraph(url=url, username=username, password=password)
    
    try:
        model_config = CONFIG['models']['embedding']
        model_name = model_config['name']
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side=model_config['padding_side']
        )
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=model_config['trust_remote_code'], 
            torch_dtype=getattr(torch, model_config['torch_dtype'])
        )

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        # Stage 1: Generate Cypher query from LLM response
        result_1 = stage_1(graph, question, model, tokenizer)
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if not result_1:
            print("Stage 1 failed ")
        else:
            # print("Stage 1 results:", result_1)
            print("Stage 1 passed")
            return result_1
        # embedding_model = SentenceTransformer("intfloat/e5-large-v2")

        # question_embedding = create_embedding(question, embedding_model=embedding_model)
        # result_2 = stage_2(graph, question_embedding, question)
        # if not result_2:
        #     print("Stage 2 failed")
        #     result_3 = stage_3(graph, embedding_model, question)
        #     if not result_3:
        #         print("Stage 3 failed")
        #     else:
        #         print("Stage 3 results ", result_3)
        # else:
        #     print("Stage 2 results:", result_2)

        
        return None

    finally:
        # Always close the graph connection
        try:
            graph.close()
            print("✓ Graph connection closed successfully")
        except Exception as e:
            print(f"⚠️ Warning: Error closing graph connection: {e}")


if __name__ == "__main__":
    question = "Tell me about the agent module and its capabilities."
    main(question=question)
