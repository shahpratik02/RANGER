import json
import os
from pathlib import Path
from time import time
from tqdm import tqdm
import yaml
from typing import Any, Dict, List, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
from langchain_neo4j import Neo4jGraph
from transformers import AutoTokenizer
from tiktoken import get_encoding
import torch
from src.utils.code_2_text_prompts import summarisation_prompt, members_prompt, file_prompt
from vllm import LLM, SamplingParams
import openai
import httpx
import concurrent.futures
import argparse
encoding = get_encoding("cl100k_base")

def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Count tokens in text. Uses provided tokenizer if available, otherwise tiktoken."""
    if tokenizer:
        # Use the provided tokenizer (e.g., from Hugging Face)
        return len(tokenizer.encode(text))
    try:
        # Fallback to tiktoken for approximation (e.g., for remote OpenAI models)
        return len(encoding.encode(text))
    except Exception as e:
        # Final fallback for any unexpected errors
        return len(text) // 4

def truncate_text_by_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens using a specific tokenizer."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        # Decode back to string, preserving space formatting for code
        return tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    return text

def format_descriptions_for_summarization(descriptions: List[str]) -> str:
    """Format descriptions with bold numbering and proper spacing for summarization."""
    if not descriptions:
        return ""
    
    formatted_parts = []
    for i, desc in enumerate(descriptions, 1):
        formatted_parts.append(f"**{i}. {desc}")
    
    return "\n\n".join(formatted_parts)

def check_context_fit(code: str, prompts: Dict, max_tokens: int = 7000, tokenizer: Optional[Any] = None) -> bool:
    """Check if code + prompts will fit within context length."""
    system_prompts = [
        prompts["base_prompt"],
        prompts["object_prompt"]
    ]
    max_formatted_tokens = 0
    for system_prompt in system_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Count actual tokens in formatted prompt
        formatted_tokens = len(tokenizer.encode(formatted_prompt))
        max_formatted_tokens = max(max_formatted_tokens, formatted_tokens)
    
    # Add response buffer
    response_buffer = 3000
    total_tokens = max_formatted_tokens + response_buffer
    
    return total_tokens <= max_tokens

def chunk_descriptions_by_tokens(descriptions: List[str], max_tokens: int = 7000) -> List[List[str]]:
    """Split descriptions into chunks that fit within token limit."""
    if not descriptions:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for desc in descriptions:
        desc_tokens = count_tokens(desc)
        
        if desc_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            # Split large description by sentences
            sentences = desc.split(". ")
            temp_chunk = []
            temp_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                if temp_tokens + sentence_tokens > max_tokens and temp_chunk:
                    chunks.append([". ".join(temp_chunk) + "."])
                    temp_chunk = [sentence]
                    temp_tokens = sentence_tokens
                else:
                    temp_chunk.append(sentence)
                    temp_tokens += sentence_tokens
            
            if temp_chunk:
                chunks.append([". ".join(temp_chunk)])
        
        elif current_tokens + desc_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [desc]
            current_tokens = desc_tokens
        else:
            current_chunk.append(desc)
            current_tokens += desc_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def recursive_summarize(
    descriptions: List[str],
    code2json: "Code2Json",
    config: Dict,
    max_tokens: int = 7000,
    max_depth: int = 3,
    current_depth: int = 0,
    **kwargs,
) -> str:

    """Recursively summarize descriptions if they exceed token limit."""
    if current_depth >= max_depth:
        print(f"    Max recursion depth {max_depth} reached, truncating descriptions")
        total_text = "\n".join(descriptions)
        if count_tokens(total_text) > max_tokens:
            truncated = ""
            for desc in descriptions:
                if count_tokens(truncated + desc) > max_tokens:
                    break
                truncated += desc + "\n"
            return truncated.strip()
        return total_text
    
    combined_descriptions = format_descriptions_for_summarization(descriptions)
    total_tokens = count_tokens(combined_descriptions)
    
    print(f"    Recursion depth {current_depth}: {len(descriptions)} descriptions, {total_tokens} tokens")
    
    if total_tokens <= max_tokens:
        return summarize_descriptions_direct(descriptions, code2json, config, **kwargs)
    
    chunks = chunk_descriptions_by_tokens(descriptions, max_tokens)
    print(f"    Split into {len(chunks)} chunks for processing")
    
    # Prepare all queries for batch processing
    chunk_queries = []
    for chunk in chunks:
        descriptions_text = format_descriptions_for_summarization(chunk)
        system_prompt = config["PROMPTS"]["file_prompt"]
        user_prompt = descriptions_text 
        
        query = {
            "url": code2json.REMOTE_URL,
            "model": code2json.REMOTE_MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "file_name": kwargs.get("file_name", "unknown"),
            "file_path": kwargs.get("file_path", "unknown"),
            "code_preview": "File-level description (no code)",
        }
        chunk_queries.append(query)
    
    # Process all chunks in parallel
    chunk_summaries = code2json.get_remote_llm_responses_batch(chunk_queries)
    
    return recursive_summarize(
        chunk_summaries,
        code2json,
        config,
        max_tokens,
        max_depth,
        current_depth + 1,
        **kwargs,
    )

def summarize_descriptions_direct(descriptions: List[str], code2json: "Code2Json", config: Dict, **kwargs) -> str:
    """Direct summarization of descriptions using remote LLM."""
    descriptions_text = format_descriptions_for_summarization(descriptions)
    system_prompt = config["PROMPTS"]["file_prompt"]
    user_prompt = descriptions_text 
    
    query = {
        "url": code2json.REMOTE_URL,
        "model": code2json.REMOTE_MODEL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "file_name": kwargs.get("file_name", "unknown"),
        "file_path": kwargs.get("file_path", "unknown"),
        "code_preview": "File-level description (no code)",
    }
    
    return code2json.get_remote_llm_response(query)

def process_code_chunks_globally(graph: Neo4jGraph, code2json: "Code2Json", embedding_model, repo_name: str):
    """
    Process all unprocessed functions, methods, small classes, and global variables in one global batch for a specific repository.
    """
    print(f"Starting global processing of code chunks for repo: {repo_name}...")
    
    query = """
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS*]->(n)
    WHERE (n:Function OR n:Class OR n:GlobalVariable) AND n.embedding IS NULL AND n.code IS NOT NULL
    RETURN n, labels(n) AS labels
    UNION
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS*]->(:Class)-[:HAS_METHOD]->(n:Method)
    WHERE n.embedding IS NULL AND n.code IS NOT NULL
    RETURN n, labels(n) AS labels
    """
    results = graph.query(query, {"repo_name": repo_name})
    
    chunks_to_process = []
    metadata_list = []
    large_classes_to_process_later = []

    for record in tqdm(results, desc="Categorizing nodes"):
        node = record["n"]
        labels = record["labels"]
        if not check_context_fit(node["code"], code2json.PROMPTS, max_tokens=14000, tokenizer=code2json.tokenizer):
            if "Class" in labels:
                large_classes_to_process_later.append(node)
                continue
            else:
                # Precisely calculate max code tokens and truncate
                system_prompts = [code2json.PROMPTS["base_prompt"], code2json.PROMPTS["object_prompt"]]
                system_prompt_tokens = max(count_tokens(p, code2json.tokenizer) for p in system_prompts)
                
                # max_tokens=15000, response_buffer=3000, so prompt limit is 12000
                prompt_limit = 11000
                max_code_tokens = prompt_limit - system_prompt_tokens
                
                original_tokens = count_tokens(node['code'], code2json.tokenizer)
                print(f"    Warning: Node {node['name']} code is too long ({original_tokens} tokens). Truncating to {max_code_tokens} tokens.")
                
                node["code"] = truncate_text_by_tokens(node["code"], code2json.tokenizer, max_code_tokens)
                
        
            
        node_key = {
            "module_name": node["module_name"],
            "name": node["name"],
        }
        if any(label in ["Class", "Function", "Method"] for label in labels):
            node_key["signature"] = node["signature"]
            
        chunk = {
            "code": node["code"],
            "name": node["name"],
            "node_key": node_key
        }
        chunks_to_process.append(chunk)
        metadata_list.append({"node": node, "labels": labels})

    if not chunks_to_process:
        print("No simple code chunks to process.")
        return large_classes_to_process_later

    llm_results = code2json.create_data_batch(chunks_to_process)
    
    texts_for_embedding = []
    for result in llm_results:
        text = (
            (result["description"] if isinstance(result["description"], str) else "\n".join(result["description"]))
            + "\n" + "\n".join(result.get("members_descriptions", []))
        )
        texts_for_embedding.append(text)
        
    embeddings = create_embedding_batch(texts_for_embedding, embedding_model)
    
    database_updates = []
    for result, metadata, embedding in zip(llm_results, metadata_list, embeddings):
        node = metadata["node"]
        labels = metadata["labels"]
        
        combined_description = "\n".join(result.get("members_descriptions", []))
        
        update_data = {
            "name": node["name"],
            "module_name": node["module_name"],
            "embedding": embedding,
            "description": result["description"],
            "member_descriptions": combined_description,
        }
        if any(label in ["Class", "Function", "Method"] for label in labels):
            update_data["signature"] = node["signature"]
        
        database_updates.append(update_data)
        
    batch_update_nodes(graph, database_updates)
    
    print(f"Finished processing {len(database_updates)} code chunks.")
    return large_classes_to_process_later

def process_large_classes_globally(large_classes: List[Dict], graph: Neo4jGraph, code2json: "Code2Json", embedding_model, config: Dict, repo_name: str):
    if not large_classes:
        return

    print(f"Processing {len(large_classes)} large classes for repo: {repo_name}...")
    batch_size = 8
    # Prepare all large class data and queries
    large_class_data = []
    for large_class in large_classes:
        try:
            methods_query = """
            MATCH (r:Repo {name: $repo_name})-[:CONTAINS*]->(c:Class {name: $class_name, module_name: $module_name, signature: $signature})-[:HAS_METHOD]->(method)
            WHERE method.description IS NOT NULL
            RETURN method.description as description
            """
            method_results = graph.query(methods_query, {
                "repo_name": repo_name,
                "class_name": large_class["name"],
                "module_name": large_class["module_name"],
                "signature": large_class["signature"],
            })
            method_descriptions = [r['description'] for r in method_results if r['description']]

            if not method_descriptions:
                print(f"    Warning: Large class {large_class['name']} has no methods with descriptions.")
                continue

            large_class_data.append({
                "class_node": large_class,
                "method_descriptions": method_descriptions
            })
        except Exception as exc:
            print(f'Large class {large_class["name"]} generated an exception: {exc}')

    if not large_class_data:
        return

    # Separate items based on token count
    recursive_items = []
    batch_items = []
    
    for item in large_class_data:
        descriptions_text = format_descriptions_for_summarization(item["method_descriptions"])
        if count_tokens(descriptions_text) > 7000:
            recursive_items.append(item)
        else:
            batch_items.append(item)

    database_updates = []
    
    # Process recursive items individually
    if recursive_items:
        print(f"Processing {len(recursive_items)} large classes with recursive summarization...")
        for item in recursive_items:
            large_class = item["class_node"]
            method_descriptions = item["method_descriptions"]
            
            print(f"    Processing large class {large_class['name']} with recursive summarization")
            class_description = recursive_summarize(
                method_descriptions,
                code2json,
                config,
                max_tokens=7000,
                file_name=large_class["name"],
                file_path=large_class.get("module_name", ""),
            )
            
            # Create embedding and prepare update
            class_embedding = create_embedding_batch([class_description], embedding_model)[0]
            combined_method_descriptions = "\n".join(method_descriptions)
            
            class_update = {
                "name": large_class["name"],
                "module_name": large_class["module_name"],
                "signature": large_class["signature"],
                "embedding": class_embedding,
                "description": class_description,
                "member_descriptions": combined_method_descriptions,
            }
            database_updates.append(class_update)
    
    # Process batch items in parallel batches of 4
    if batch_items:
        print(f"Processing {len(batch_items)} large classes in batches...")
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(batch_items) + batch_size - 1)//batch_size}")
            
            # Prepare queries for this batch
            batch_queries = []
            for item in batch:
                descriptions_text = format_descriptions_for_summarization(item["method_descriptions"])
                system_prompt = config["PROMPTS"]["file_prompt"]
                user_prompt = descriptions_text
                
                query = {
                    "url": code2json.REMOTE_URL,
                    "model": code2json.REMOTE_MODEL,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "file_name": item["class_node"]["name"],
                    "file_path": item["class_node"].get("module_name", ""),
                    "code_preview": "File-level description (no code)",
                }
                batch_queries.append(query)
            
            # Process batch in parallel
            batch_results = code2json.get_remote_llm_responses_batch(batch_queries)
            class_embeddings = create_embedding_batch(batch_results, embedding_model)
            # Process results and prepare database updates
            for item, class_description, class_embedding in zip(batch, batch_results, class_embeddings):
                large_class = item["class_node"]
                method_descriptions = item["method_descriptions"]
                
                combined_method_descriptions = "\n".join(method_descriptions)
                

                class_update = {
                    "name": large_class["name"],
                    "module_name": large_class["module_name"],
                    "signature": large_class["signature"],
                    "embedding": class_embedding,
                    "description": class_description,
                    "member_descriptions": combined_method_descriptions,
                }
                database_updates.append(class_update)

    if database_updates:
        batch_update_nodes(graph, database_updates)

def process_modules_globally(graph: Neo4jGraph, code2json: "Code2Json", embedding_model, config: Dict, repo_name: str):
    """
    Process all modules that don't have an embedding for a specific repository.
    Assumes all children nodes have been processed and have descriptions.
    """
    print(f"Processing modules for repository: {repo_name}...")
    module_query = """
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)
    WHERE m.embedding IS NULL
    RETURN m
    """
    module_results = graph.query(module_query, {"repo_name": repo_name})
    
    if not module_results:
        print("No modules to process for this repository.")
        return
    batch_size = 8
    module_data = []
    for module_record in tqdm(module_results, desc="Gathering module data"):
        module_node = module_record["m"]
        
        children_query = """
        MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(:Module {name: $module_name})-[:CONTAINS]->(child)
        WHERE child.description IS NOT NULL
        RETURN child.description as description
        """
        children_results = graph.query(children_query, {"repo_name": repo_name, "module_name": module_node["name"]})
        
        child_descriptions = [r['description'] for r in children_results if r['description']]

        if not child_descriptions:
            print(f"Module {module_node['name']} has no children with descriptions. Skipping.")
            continue
        
        module_data.append({
            "module_node": module_node,
            "child_descriptions": child_descriptions
        })

    if not module_data:
        return

    # Separate items based on token count
    recursive_items = []
    batch_items = []
    
    for item in module_data:
        descriptions_text = format_descriptions_for_summarization(item["child_descriptions"])
        if count_tokens(descriptions_text) > 7000:
            recursive_items.append(item)
        else:
            batch_items.append(item)

    updates = []
    
    # Process recursive items individually
    if recursive_items:
        print(f"Processing {len(recursive_items)} modules with recursive summarization...")
        for item in recursive_items:
            module_node = item["module_node"]
            child_descriptions = item["child_descriptions"]
            
            print(f"    Processing module {module_node['name']} with recursive summarization")
            module_description = recursive_summarize(
                child_descriptions,
                code2json,
                config,
                max_tokens=7000,
                file_name=module_node["name"],
                file_path=module_node.get("path", ""),
            )
            
            # Create embedding and prepare update
            module_embedding = create_embedding_batch([module_description], embedding_model)[0]
            
            module_update = {
                "name": module_node["name"],
                "embedding": module_embedding,
                "description": module_description,
            }
            updates.append(module_update)
    
    # Process batch items in parallel batches of 4
    if batch_items:
        print(f"Processing {len(batch_items)} modules in batches...")
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(batch_items) + batch_size - 1)//batch_size}")
            
            # Prepare queries for this batch
            batch_queries = []
            for item in batch:
                descriptions_text = format_descriptions_for_summarization(item["child_descriptions"])
                system_prompt = config["PROMPTS"]["file_prompt"]
                user_prompt = descriptions_text
                
                query = {
                    "url": code2json.REMOTE_URL,
                    "model": code2json.REMOTE_MODEL,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "file_name": item["module_node"]["name"],
                    "file_path": item["module_node"].get("path", ""),
                    "code_preview": "File-level description (no code)",
                }
                batch_queries.append(query)
            
            # Process batch in parallel
            batch_results = code2json.get_remote_llm_responses_batch(batch_queries)
            module_embeddings = create_embedding_batch(batch_results, embedding_model)
            # Process results and prepare database updates
            for item, module_description, module_embedding in zip(batch, batch_results, module_embeddings):
                module_node = item["module_node"]
                
                
                
                module_update = {
                    "name": module_node["name"],
                    "embedding": module_embedding,
                    "description": module_description,
                }
                updates.append(module_update)

    if updates:
        print(f"Batch updating {len(updates)} modules for repo {repo_name}...")
        module_update_query = """
        UNWIND $updates AS update
        MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module {name: update.name})
        SET m.embedding = update.embedding, m.description = update.description
        """
        graph.query(module_update_query, {"updates": updates, "repo_name": repo_name})

class Code2Json:
    def __init__(self, url: str, model: str, prompts: Dict) -> None:
        self.URL = url
        self.MODEL = model
        self.PROMPTS = prompts
        
        # Initialize vLLM
        self.llm = LLM(
            model="deepseek-ai/deepseek-coder-1.3b-instruct",
            tensor_parallel_size=1,  # Adjust based on your GPU setup
            gpu_memory_utilization=0.8,
            max_model_len=16384,
            trust_remote_code=True
        )
        
        # Initialize tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        
        # Sampling parameters for vLLM
        self.sampling_params = SamplingParams(
            temperature=0.01,
            max_tokens=3072,
            stop=["</s>", "<|im_end|>"]
        )

        # Remote endpoint for large classes and modules
        self.REMOTE_URL = "http://localhost:8000/v1"  # Update with your vLLM endpoint URL
        self.REMOTE_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

        # Create a reusable client for the remote LLM
        api_key = "EMPTY"
        timeout = httpx.Timeout(
            connect=30.0,  # 30 seconds to establish connection
            read=300.0,    # 5 minutes to read response
            write=30.0,    # 30 seconds to send request
            pool=30.0      # 30 seconds to get connection from pool
        )
        self.remote_client = openai.OpenAI(
            api_key=api_key,
            base_url=self.REMOTE_URL,
            http_client=httpx.Client(verify=False, timeout=timeout)
        )

    def get_remote_llm_response(self, query: Dict) -> str:
        """Generate text using remote OpenAI-compatible endpoint."""
        model = query["model"]

        try:
            stream = self.remote_client.chat.completions.create(
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
            gen_tokens = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    gen_tokens += content
            return gen_tokens.strip()
        except Exception as e:
            print(f"Error in remote OpenAI API call: {e}")
            print(f"File: {query['file_name']}")
            print(f"Code preview:\n{query['code_preview']}")
            return ""

    def get_remote_llm_responses_batch(self, queries: List[Dict]) -> List[str]:
        """Process multiple queries using remote OpenAI-compatible endpoint in parallel."""
        if not queries:
            return []
        
        print(f"Processing {len(queries)} queries with remote endpoint in batch...")
        
        def process_single_query(query):
            return self.get_remote_llm_response(query)
        
        # Use ThreadPoolExecutor for parallel processing with 4 workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_single_query, queries))
        
        return results

    def get_llm_response(self, query: Dict) -> str:
        """Generate text using vLLM."""
        try:
            messages = [
                {"role": "system", "content": query["system_prompt"]},
                {"role": "user", "content": query["user_prompt"]}
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = self.llm.generate([formatted_prompt], self.sampling_params)
            
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            print(f"Error in vLLM inference: {e}")
            print(f"File: {query.get('file_name', 'unknown')}")
            return ""

    def get_llm_responses_batch(self, queries: List[Dict]) -> List[str]:
        """Process multiple queries using vLLM's built-in batching."""
        if not queries:
            return []
        
        try:
            # Prepare all prompts
            formatted_prompts = []
            for query in queries:
                messages = [
                    {"role": "system", "content": query["system_prompt"]},
                    {"role": "user", "content": query["user_prompt"]}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            
            # Generate batch with vLLM
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            
            return [output.outputs[0].text.strip() for output in outputs]
            
        except Exception as e:
            print(f"Error in batch vLLM inference: {e}")
            return [self.get_llm_response(query) for query in queries]
    def postprocess_descriptions(self, path: str, description: str, include_name: str = "", null_delimiter: str = "---None---") -> Tuple[List[str], List[str]]:
        """Format the LLM generated descriptions into an itemized list format."""
        if include_name is None:
            pathless_description = [
                item for item in description.split("\n") if item != null_delimiter
            ]
        else:
            pathless_description = list(
                map(
                    lambda x: f"{include_name} - " + x,
                    [item for item in description.split("\n") if item != null_delimiter],
                )
            )
        
        path_description = [
            "path: " + str(path) + " - " + x for x in pathless_description
        ]
        
        return pathless_description, path_description

    def get_llm_descriptions(self, url: str, model: str, code: str, file_path: str, prompts: Dict, language: str, **kwargs) -> Dict:
        """Generate various functional descriptions for the given code using an LLM."""
        descriptions = {}
        
        system_prompts = [
            prompts["base_prompt"],
            prompts["object_prompt"]
        ]
        
        user_prompt = code 
        
        # Create queries for batch processing
        queries = [
            {
                "system_prompt": system_prompts[i],
                "user_prompt": user_prompt,
                "file_name": kwargs.get("file_name", "unknown"),
                "file_path": file_path,
                "code_preview": "\n".join(code.split("\n")[:5]),
            }
            for i in range(len(system_prompts))
        ]
        
        # Get batch responses
        responses = self.get_llm_responses_batch(queries)
        
        descriptions["description"] = responses[0]
        members_response = responses[1]
        
        obj_descriptions = self.postprocess_descriptions(
            path=file_path, description=members_response
        )
        
        if obj_descriptions is not None:
            descriptions["members_descriptions"], descriptions["path_members_descriptions"] = obj_descriptions
        else:
            descriptions["members_descriptions"] = []
            descriptions["path_members_descriptions"] = []
        
        return descriptions

    def get_llm_descriptions_file(self, url: str, model: str, prompts: Dict, descriptions: List[str], max_tokens: int = 7000, use_recursive_summarization: bool = True, **kwargs) -> Dict:
        """Generate a description for the entire file using remote LLM."""
        # Use remote endpoint for file-level processing
        config = {
            "URL": self.REMOTE_URL,
            "MODEL": self.REMOTE_MODEL,
            "PROMPTS": prompts,
        }
        
        descriptions_dict = {}
        
        if use_recursive_summarization:
            descriptions_dict["description"] = recursive_summarize(
                descriptions, self, config, max_tokens=max_tokens, **kwargs, current_depth=0
            )
        else:
            descriptions_text = format_descriptions_for_summarization(descriptions)
            system_prompt = prompts["file_prompt"]
            user_prompt = descriptions_text 
            
            query = {
                "url": self.REMOTE_URL,
                "model": self.REMOTE_MODEL,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "file_name": kwargs.get("file_name", "unknown"),
                "file_path": kwargs.get("file_path", "unknown"),
                "code_preview": "File-level description (no code)",
            }
            
            descriptions_dict["description"] = self.get_remote_llm_response(query)
        
        descriptions_dict["members_descriptions"] = None
        descriptions_dict["path_members_descriptions"] = None
        
        return descriptions_dict

    def create_data_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Process multiple chunks - vLLM handles batching internally."""
        print(f"Processing {len(chunks)} chunks in a batch...")
        if not chunks:
            return []

        # 1. Prepare all queries from all chunks
        all_queries = []
        chunk_info = []
        for chunk in chunks:
            chunk_code = chunk["code"]
            user_prompt = chunk_code 
            system_prompts = [
                self.PROMPTS["base_prompt"],
                self.PROMPTS["object_prompt"],
            ]
            
            chunk_info.append({
                "name": chunk["name"],
                "node_key": chunk["node_key"],
                "code": chunk["code"]
            })
            
            for system_prompt in system_prompts:
                query = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "file_name": chunk["name"],
                    "file_path": chunk["node_key"],
                    "code_preview": "\n".join(chunk_code.split("\n")[:5]),
                }
                all_queries.append(query)

        # 2. Get all LLM responses in one batch
        all_responses = self.get_llm_responses_batch(all_queries)

        if len(all_responses) != len(all_queries):
            print(
                f"Error: Mismatch in number of responses ({len(all_responses)}) and queries ({len(all_queries)})."
            )
            return []

        # 3. Process responses and map them back to chunks
        results = []
        num_prompts_per_chunk = 2
        for i, chunk_data in enumerate(chunk_info):
            start_index = i * num_prompts_per_chunk
            end_index = (i + 1) * num_prompts_per_chunk
            if end_index > len(all_responses):
                print(f"Warning: Not enough responses for chunk {chunk_data['name']}. Skipping.")
                continue

            chunk_responses = all_responses[start_index:end_index]
            
            description = chunk_responses[0]
            members_response = chunk_responses[1]
            
            obj_descriptions = self.postprocess_descriptions(
                path=chunk_data["node_key"], description=members_response
            )
            
            if obj_descriptions is not None:
                members_descriptions, path_members_descriptions = obj_descriptions
            else:
                members_descriptions = []
                path_members_descriptions = []

            formatted_chunk = {
                "file_name": chunk_data["name"],
                "file_path": chunk_data["node_key"],
                "raw_code": chunk_data["code"],
                "description": description,
                "members_descriptions": members_descriptions,
                "path_members_descriptions": path_members_descriptions,
            }
            results.append(formatted_chunk)
            print(f"Processed {chunk_data['name']}, Data Created!")
            
        return results


def create_embedding_batch(texts: List[str], embedding_model) -> List:
    """Create embeddings for multiple texts in a single batch."""
    if not texts:
        return []
    
    print(f"Creating embeddings for {len(texts)} texts in batch...")
    embeddings = embedding_model.encode(texts, batch_size=16, show_progress_bar=True)
    return embeddings.tolist()

def batch_update_nodes(graph: Neo4jGraph, updates: List[Dict]) -> None:
    """Update multiple nodes in a single transaction."""
    if not updates:
        return
    
    print(f"Batch updating {len(updates)} nodes...")
    
    signature_updates = []
    name_only_updates = []
    
    for update in updates:
        if "signature" in update:
            signature_updates.append(update)
        else:
            name_only_updates.append(update)
    
    if signature_updates:
        batch_query = """
        UNWIND $updates AS update
        MATCH (n {name: update.name, module_name: update.module_name, signature: update.signature})
        SET n.embedding = update.embedding, 
            n.description = update.description, 
            n.member_descriptions = update.member_descriptions
        """
        graph.query(batch_query, {"updates": signature_updates})
    
    if name_only_updates:
        batch_query = """
        UNWIND $updates AS update
        MATCH (n {name: update.name, module_name: update.module_name})
        SET n.embedding = update.embedding, 
            n.description = update.description, 
            n.member_descriptions = update.member_descriptions
        """
        graph.query(batch_query, {"updates": name_only_updates})

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a code repository in the graph database.")
    parser.add_argument("repo_name", help="The name of the repository to process (must match the 'name' property of the :Repo node).")
    args = parser.parse_args()
    repo_name = args.repo_name

    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "your_neo4j_password"  # Update with your Neo4j password
    
    with open("data_prep.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    language_path = config.get("language_path")
    
    PROMPTS = {
        "base_prompt": summarisation_prompt,
        "file_prompt": file_prompt,
        "object_prompt": members_prompt
    }
    
    URL = "local"
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    # GPU-optimized embedding model
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    embedding_model.to('cuda')  # Ensure it runs on GPU
    
    config_dict = {
        "URL": URL,
        "MODEL": MODEL,
        "PROMPTS": PROMPTS,
    }
    
    code2json = Code2Json(url=URL, model=MODEL, prompts=PROMPTS)
    graph = Neo4jGraph(url=url, username=username, password=password)
    
    # New global processing flow
    print(f"Starting new global processing flow for repository: {repo_name}...")

    # 1. Process all atomic code chunks (functions, methods, small classes)
    large_classes = process_code_chunks_globally(graph, code2json, embedding_model, repo_name)

    # 2. Process large classes now that their methods should have descriptions
    process_large_classes_globally(large_classes, graph, code2json, embedding_model, config_dict, repo_name)

    # 3. Process modules now that all their children should have descriptions
    process_modules_globally(graph, code2json, embedding_model, config_dict, repo_name)
    
    print("Deleting empty modules...")
    delete_query = """
    MATCH (r:Repo {name: $repo_name})-[:CONTAINS]->(m:Module)
    WHERE NOT EXISTS((m)-[:CONTAINS]->())
    DETACH DELETE m
    """
    graph.query(delete_query, {"repo_name": repo_name})
    print("Deleted empty modules")
    
    graph.close()
    print("Processing complete!")

if __name__ == "__main__":
    main()