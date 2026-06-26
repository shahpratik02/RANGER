import os
import json
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher, ndiff
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import torch
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch.multiprocessing as mp

from src.core.retirever_v2 import stage_1
from src.utils.config_loader import CONFIG
from pathlib import Path
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi

# Global vLLM model instance
GLOBAL_LLM = None
GLOBAL_TOKENIZER = None
import multiprocessing

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# def initialize_global_model():
#     """Initialize the global vLLM model and tokenizer"""
#     global GLOBAL_LLM, GLOBAL_TOKENIZER
#     if GLOBAL_LLM is None:
#         print("Initializing global vLLM model: deepseek-ai/deepseek-coder-1.3b-base")
#         GLOBAL_LLM = LLM(
#             model="deepseek-ai/deepseek-coder-1.3b-base",
#             gpu_memory_utilization=0.8,  # Use 80% of GPU memory         # Limit context length
#             tensor_parallel_size=1,
#         )
#         GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
#             "deepseek-ai/deepseek-coder-1.3b-base"


class PythonCodeBM25Searcher:
    """
    Simple BM25 index for Python code files using rank-bm25 library
    """

    def __init__(self, folder_path: str, max_lines_per_chunk: int = 10):
        """
        Initialize the BM25 searcher

        Args:
            folder_path: Path to folder containing Python files
            max_lines_per_chunk: Maximum lines per chunk (default: 10)
        """
        self.folder_path = Path(folder_path)
        self.max_lines_per_chunk = max_lines_per_chunk
        self.corpus = []
        self.chunk_metadata = []  # Store file info for each chunk
        self.bm25 = None

        # Build index on initialization
        self._build_index()

    def _get_python_files(self) -> List[Path]:
        """Get all Python files from the folder recursively"""
        return list(self.folder_path.rglob("*.py"))

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that splits on whitespace and removes special chars"""
        import string

        # Remove punctuation and split on whitespace
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.lower().split()

    def _chunk_code(self, code: str, file_path: Path) -> List[Dict]:
        """
        Chunk Python code into max_lines_per_chunk line segments

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of chunk dictionaries with metadata
        """
        lines = code.split("\n")
        chunks = []

        for i in range(0, len(lines), self.max_lines_per_chunk):
            chunk_lines = lines[i : i + self.max_lines_per_chunk]
            chunk_text = "\n".join(chunk_lines).strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    {
                        "text": chunk_text,
                        "file_path": str(file_path),
                        "start_line": i + 1,
                        "end_line": min(i + self.max_lines_per_chunk, len(lines)),
                        "total_lines": len(lines),
                    }
                )

        return chunks

    def _build_index(self):
        """Build BM25 index from all Python files in the folder"""
        python_files = self._get_python_files()

        if not python_files:
            raise ValueError(f"No Python files found in {self.folder_path}")

        print(f"Found {len(python_files)} Python files. Building index...")

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                # Chunk the code
                chunks = self._chunk_code(code, file_path)

                # Add chunks to corpus
                for chunk in chunks:
                    self.corpus.append(chunk["text"])
                    self.chunk_metadata.append(chunk)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not self.corpus:
            raise ValueError("No code chunks were created")

        print(f"Created {len(self.corpus)} code chunks. Creating BM25 index...")

        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("BM25 index created successfully!")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for most relevant code chunks

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing chunk info and scores
        """
        if not self.bm25:
            raise ValueError("Index not built yet")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices and scores
        top_indices = scores.argsort()[-top_k:][::-1]

        search_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                result = {
                    "score": float(scores[idx]),
                    "text": self.corpus[idx],
                    "metadata": self.chunk_metadata[idx],
                }
                search_results.append(result)

        return search_results

    def search_excluding_file(
        self, query: str, exclude_file: str, top_k: int = 5
    ) -> List[Dict]:
        """
        Search for most relevant code chunks while excluding chunks from a specific file

        Args:
            query: Search query
            exclude_file: File path to exclude from results
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing chunk info and scores
        """
        if not self.bm25:
            raise ValueError("Index not built yet")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get more candidates to filter from
        top_indices = scores.argsort()[-top_k * 3 :][::-1]

        search_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                # Skip chunks from the excluded file
                chunk_file = self.chunk_metadata[idx]["file_path"]

                chunk_normalized_file = Path(chunk_file).as_posix()
                exclude_normalized = Path(exclude_file).as_posix()
                if not chunk_normalized_file.endswith(exclude_normalized):
                    result = {
                        "score": float(scores[idx]),
                        "text": self.corpus[idx],
                        "metadata": self.chunk_metadata[idx],
                    }
                    search_results.append(result)
                    if len(search_results) >= top_k:
                        break

        return search_results

    def clear_index(self):
        """
        Clear the built BM25 index and reset all data structures
        """
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None
        print("BM25 index cleared successfully!")


#         )
# def initialize_global_model():
#     """Initialize the global vLLM model and tokenizer with BF16 precision"""
#     global GLOBAL_LLM, GLOBAL_TOKENIZER


#     if GLOBAL_LLM is None:
#         print("Initializing global vLLM model: Qwen/Qwen2.5-Coder-7B")
#         GLOBAL_LLM = LLM(
#             model="Qwen/Qwen2.5-Coder-7B",
#             gpu_memory_utilization=0.9,  # Use 90% of GPU memory
#             tensor_parallel_size=1,
#             dtype="bfloat16",
#             max_model_len=16384,
#             trust_remote_code=False,
#         )
#         GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
#             "Qwen/Qwen2.5-Coder-7B", trust_remote_code=False
#         )
def initialize_global_model():
    """Initialize the global vLLM model and tokenizer with BF16 precision"""
    global GLOBAL_LLM, GLOBAL_TOKENIZER

    if GLOBAL_LLM is None:
        print("Initializing global vLLM model: bigcode/starcoderbase-7b")
        GLOBAL_LLM = LLM(
            model="bigcode/starcoderbase-7b",
            gpu_memory_utilization=0.3,  # Use 30% of GPU memory
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=8192,  # StarCoder has 8k context limit
            trust_remote_code=True,  # StarCoder requires trust_remote_code=True
        )
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
            "bigcode/starcoderbase-7b", trust_remote_code=True
        )


def is_syntactically_correct(code: str) -> bool:
    """
    Check if the code is syntactically correct using AST

    Args:
        code: Python code string to check

    Returns:
        bool: True if syntactically correct, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        int: Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_exact_match(predicted: str, groundtruth: str) -> bool:
    """
    Compute Exact Match (EM) between predicted and groundtruth strings.

    Args:
        predicted: The predicted string
        groundtruth: The groundtruth string

    Returns:
        bool: True if exact match, False otherwise
    """
    return predicted.strip() == groundtruth.strip()


def compute_edit_similarity(predicted: str, groundtruth: str) -> float:
    """
    Compute Edit Similarity (ES) using the formula: 1 - Lev(Y,Y_hat)/max(|Y|,|Y_hat|)

    Args:
        predicted: The predicted string (Y_hat)
        groundtruth: The groundtruth string (Y)

    Returns:
        float: Edit similarity score between 0.0 and 1.0
    """
    pred_clean = predicted.strip()
    gt_clean = groundtruth.strip()

    # Handle edge case where both strings are empty
    if len(pred_clean) == 0 and len(gt_clean) == 0:
        return 1.0

    # Handle edge case where one string is empty
    if len(pred_clean) == 0 or len(gt_clean) == 0:
        return 0.0

    # Compute Levenshtein distance
    lev_dist = levenshtein_distance(pred_clean, gt_clean)

    # Compute edit similarity: 1 - Lev(Y,Y_hat)/max(|Y|,|Y_hat|)
    max_len = max(len(pred_clean), len(gt_clean))
    edit_similarity = 1.0 - (lev_dist / max_len)

    return max(0.0, edit_similarity)  # Ensure non-negative result


def format_retrieved_context(
    retrieved_results: List[Dict],
    bm25_retrieved_results: List[str],
    code_tokenizer,
    prompt: str,
    max_context_tokens: int = CONFIG["processing"]["max_context_tokens"],
) -> str:
    """
    Formats retrieved code structure as comments and prepends to the prompt,
    respecting a token limit for the context part only.

    The format is:
    # Module: <module_name>
    #   Class: <class_name>
    #     - <method_signature>
    #   Functions:
    #     - <function_signature>

    Args:
        retrieved_results: List of retrieved code entities.
        bm25_retrieved_results: List of BM25 retrieved code chunks (strings).
        code_tokenizer: Tokenizer to count tokens.
        prompt: The original prompt.
        max_context_tokens: Maximum tokens for the generated context comments.
                            The prompt's own token count is not part of this limit.

    Returns:
        String with retrieved context as comments prepended to the prompt.
    """
    context_parts = []
    total_tokens_used = 0

    # 1. Process main retrieved_results (limit: 3000 tokens)
    current_tokens = 0
    if retrieved_results:
        # Group results by module and then by class
        grouped = defaultdict(lambda: defaultdict(list))
        for item in retrieved_results:
            module_name = item.get("module_name", "unknown_module")
            class_name = item.get("class")
            if class_name:
                grouped[module_name][class_name].append(item)
            else:
                grouped[module_name]["<standalone_functions>"].append(item)

        grouped_results = {k: dict(v) for k, v in grouped.items()}

        # Build the main context string, respecting the 3000 token limit
        header = "# Here is a summary of relevant code from other files:"
        main_context_parts = [header]

        current_tokens = len(code_tokenizer.encode(header))
        max_main_tokens = 3000  # Fixed at 3000 tokens

        for module_name, entities in grouped_results.items():
            # Create the string for the current module block
            module_block_lines = [f"\n# Module: {module_name}"]

            sorted_entities = sorted(
                entities.items(), key=lambda item: item[0] == "<standalone_functions>"
            )

            for entity_name, items in sorted_entities:
                if entity_name == "<standalone_functions>":
                    if not items:
                        continue
                    module_block_lines.append(f"#   Functions/Classes:")
                    for func in items:
                        display = func.get("signature", "")
                        if display is None or len(display) == 0:
                            display = func.get("name", "N/A")
                        module_block_lines.append(f"#     - {display}")
                else:  # It's a class
                    module_block_lines.append(f"#   Class: {entity_name}")
                    for method in items:
                        display = method.get("signature", "")
                        if display is None or len(display) == 0:
                            display = method.get("name", "N/A")
                        module_block_lines.append(f"#     - {display}")

            module_block = "\n".join(module_block_lines)
            module_tokens = len(code_tokenizer.encode(module_block))

            # Check if adding this module block would exceed the main context limit
            if current_tokens + module_tokens > max_main_tokens:
                print(
                    f"Main context token limit ({max_main_tokens}) reached. Stopping main context generation."
                )
                break

            main_context_parts.append(module_block)
            current_tokens += module_tokens

        if len(main_context_parts) > 1:  # More than just header was added
            context_parts.append("".join(main_context_parts))
            total_tokens_used = current_tokens

    # 2. Process BM25 retrieved_results - dynamic limit based on remaining tokens
    if bm25_retrieved_results:
        # Calculate remaining tokens for BM25, with a minimum of 1000 if retrieved_results hit limit
        remaining_tokens = max_context_tokens - total_tokens_used
        max_bm25_tokens = (
            min(1000, remaining_tokens) if current_tokens >= 3000 else remaining_tokens
        )

        bm25_header = "\n\n# Here are some structurally same code fragments:"
        bm25_context_parts = [bm25_header]

        bm25_current_tokens = len(code_tokenizer.encode(bm25_header))

        for code_text in bm25_retrieved_results:
            # Convert the code chunk to comments (prefix each line with #)
            code_lines = code_text.split("\n")
            commented_code_lines = [f"# {line}" for line in code_lines]
            commented_code = "\n".join(commented_code_lines)

            code_block = f"\n{commented_code}" + "\n#"
            code_block_tokens = len(code_tokenizer.encode(code_block))

            # Check if adding this code block would exceed the BM25 context limit
            if bm25_current_tokens + code_block_tokens > max_bm25_tokens:
                print(
                    f"BM25 context token limit ({max_bm25_tokens}) reached. Stopping BM25 context generation."
                )
                break

            bm25_context_parts.append(code_block)
            bm25_current_tokens += code_block_tokens

        if len(bm25_context_parts) > 1:  # More than just header was added
            context_parts.append("".join(bm25_context_parts))

    # 3. Combine all context parts and original prompt
    # system_prompt = "# We will give you some relevant cross-file context in comments, and you are supposed to complete the line of the incomplete code in the prompt's last line."

    if not context_parts:
        return prompt

    context_str = "".join(context_parts)
    return context_str + "\n\n" + prompt


def smart_truncate_for_starcoder(prompt: str, max_tokens: int, tokenizer) -> str:
    """
    Smart truncation that preserves code structure for StarCoder.
    Keeps imports and current function/class context intact.
    
    Args:
        prompt: The enhanced prompt to truncate
        max_tokens: Maximum tokens allowed
        tokenizer: Tokenizer to count tokens
        
    Returns:
        str: Smartly truncated prompt that preserves code structure
    """
    lines = prompt.split('\n')
    
    # Always keep imports at the beginning
    import_lines = []
    context_lines = []
    
    for line in lines:
        if 'import ' in line or 'from ' in line:
            import_lines.append(line)
        else:
            context_lines.append(line)
    
    # Find the current function/class context (work backwards from end)
    context_start = -1
    for i in range(len(context_lines) - 1, -1, -1):
        line = context_lines[i].strip()
        if line.startswith(('def ', 'class ', 'async def ')):
            context_start = i
            break
    
    # Build prompt: imports + current context + surrounding code
    if context_start != -1:
        # Take current function/class + some surrounding context (5 lines before)
        context_start_with_buffer = max(0, context_start - 5)
        relevant_context = context_lines[context_start_with_buffer:] 
    else:
        # No clear context found, take the last 50 lines
        relevant_context = context_lines[-50:] if len(context_lines) > 50 else context_lines
    
    # Combine imports + context
    candidate_lines = import_lines + relevant_context
    candidate_prompt = '\n'.join(candidate_lines)
    
    # Check if it fits
    if len(tokenizer.encode(candidate_prompt, add_special_tokens=False)) <= max_tokens:
        return candidate_prompt
    
    # If still too long, progressively remove context from the beginning
    # But always keep imports and the last part (most relevant for completion)
    min_context_lines = 10  # Always keep at least last 10 lines
    
    while len(tokenizer.encode('\n'.join(import_lines + relevant_context), add_special_tokens=False)) > max_tokens:
        if len(relevant_context) > min_context_lines:
            relevant_context.pop(0)  # Remove from beginning
        else:
            # If we can't remove more context, remove some imports if necessary
            if import_lines and len(import_lines) > 3:  # Keep at least 3 imports
                import_lines.pop(0)
            else:
                break  # Can't truncate further safely
    
    final_prompt = '\n'.join(import_lines + relevant_context)
    return final_prompt


def generate_with_code_model(
    prompt: str, max_new_tokens: int = CONFIG["processing"]["max_new_tokens"]
) -> str:
    """
    Generate code using StarCoder with FIM (Fill-in-the-Middle) prompting.
    Uses dynamic token management to fit within model limits.

    Args:
        prompt: The enhanced prompt (context + incomplete code)
        max_new_tokens: Maximum tokens to generate

    Returns:
        str: Generated code (first line)
    """
    global GLOBAL_LLM, GLOBAL_TOKENIZER

    # Initialize model if not already done
    if GLOBAL_LLM is None:
        initialize_global_model()

    # Store original prompt for syntax checking
    original_prompt = prompt
    
    # Dynamic token management for StarCoder
    max_model_tokens = CONFIG["processing"]["max_model_len"]  # 8192
    fim_overhead = 20  # Account for <fim_prefix><fim_suffix><fim_middle> tokens
    
    # Calculate maximum tokens available for the enhanced prompt
    max_prompt_tokens = max_model_tokens - max_new_tokens - fim_overhead
    
    # Check current prompt size
    prompt_tokens = GLOBAL_TOKENIZER.encode(prompt, add_special_tokens=False)
    current_prompt_length = len(prompt_tokens)
    
    if current_prompt_length > max_prompt_tokens:
        # Smart truncation: preserve code structure for StarCoder
        truncated_prompt = smart_truncate_for_starcoder(prompt, max_prompt_tokens, GLOBAL_TOKENIZER)
        
        final_tokens = len(GLOBAL_TOKENIZER.encode(truncated_prompt, add_special_tokens=False))
        print(f"🔧 Smart truncation: {current_prompt_length} -> {final_tokens} tokens")
        print(f"   Available for prompt: {max_prompt_tokens}, FIM overhead: {fim_overhead}, Generation: {max_new_tokens}")
        
        # Use truncated prompt for generation
        generation_prompt = truncated_prompt
    else:
        print(f"✅ Prompt fits: {current_prompt_length}/{max_prompt_tokens} tokens")
        generation_prompt = prompt

    # Create FIM prompt for StarCoder using the (possibly truncated) generation prompt
    # Format: <fim_prefix>prefix_code<fim_suffix>suffix_code<fim_middle>
    fim_prompt = f"<fim_prefix>{generation_prompt}<fim_suffix><fim_middle>"
    
    # Final safety check
    final_prompt_tokens = len(GLOBAL_TOKENIZER.encode(fim_prompt))
    total_budget = final_prompt_tokens + max_new_tokens
    
    if total_budget > max_model_tokens:
        print(f"⚠️  Warning: Total tokens ({total_budget}) exceeds model limit ({max_model_tokens})")
    else:
        print(f"✅ Token budget: {final_prompt_tokens} prompt + {max_new_tokens} generation = {total_budget}/{max_model_tokens}")

    # Generate with StarCoder-specific parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=CONFIG["processing"]["max_new_tokens"],
        top_p=0.95,
        top_k=20,
        stop=["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "\n\n", "<|endoftext|>"],
    )

    outputs = GLOBAL_LLM.generate([fim_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    if not generated_text:
        print("No generated text")
        return ""

    # Tokenize the generated text to get individual tokens
    generated_tokens = GLOBAL_TOKENIZER.encode(generated_text, add_special_tokens=False)

    if not generated_tokens:
        print("No generated tokens")
        return ""

    # Find positions where newlines occur
    newline_positions = []
    for i, token_id in enumerate(generated_tokens):
        token_text = GLOBAL_TOKENIZER.decode([token_id], skip_special_tokens=True)
        if "\n" in token_text:
            newline_positions.append(i)

    # If no newlines found, use all tokens
    if not newline_positions:
        newline_positions = [len(generated_tokens) - 1]

    # Try chunks up to each newline position
    for newline_pos in newline_positions:
        # Take tokens up to and including this newline position
        chunk_tokens = generated_tokens[: newline_pos + 1]

        # Decode current chunk
        current_generated = GLOBAL_TOKENIZER.decode(
            chunk_tokens, skip_special_tokens=True
        )
        # Use ORIGINAL full prompt for syntax checking (not truncated)
        full_code = original_prompt + current_generated

        # Check if current code is syntactically correct
        if is_syntactically_correct(full_code):
            # Return only the first line of generated content
            lines = current_generated.split("\n")
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:  # Return first non-empty line
                    print(
                        f"Returning syntactically correct line: {repr(stripped_line)}"
                    )
                    return stripped_line

    # If no syntactically correct chunk found, return first line of full generation
    final_generated = GLOBAL_TOKENIZER.decode(
        generated_tokens, skip_special_tokens=True
    )
    print(f"##################Final generated: {final_generated}")
    lines = final_generated.split("\n")
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # Return first non-empty line
            print(f"##################Generated code: {stripped_line}")
            return stripped_line
    fallback_result = final_generated.strip()
    print(f"##################Fallback result: {repr(fallback_result)}")
    return fallback_result


def evaluate_retrieval(
    repo_path: str,
    dataset: List[Dict],
    query_format: str,
    repo_filter: str,
    retriever_type: str,
    repo_name_for_semantic: str,
    model,  # retrieval model
    tokenizer,  # retrieval tokenizer
    neo4j_config: Dict,
    graph,
    max_examples: Optional[int] = None,
    max_total_tokens: int = 4096,
    bm25_searcher: PythonCodeBM25Searcher = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval system for code completion task.

    Args:
        repo_path: Path to the repository
        dataset: List of dictionaries with keys ['prompt', 'groundtruth', 'right_context', 'metadata']
        query_format: Format string for queries
        repo_filter: Filter by specific repository
        retriever_type: Type of retriever to use
        repo_name_for_semantic: Repository name for semantic retrieval
        model: Retrieval model (for embeddings)
        tokenizer: Retrieval tokenizer
        neo4j_config: Neo4j configuration
        graph: Neo4j graph instance for retrieval
        max_examples: Maximum number of examples to evaluate
        max_total_tokens: Maximum total tokens for enhanced prompt

    Returns:
        dict: Evaluation results with summary and detailed results
    """

    # Initialize global model if needed
    print("Initializing global model")
    initialize_global_model()
    print("Global model initialized")

    results = []
    total_examples = 0
    exact_matches = 0
    edit_similarities = []

    # Filter dataset if needed
    examples = dataset
    if repo_filter:
        examples = [
            item for item in dataset if item["metadata"]["repository"] == repo_filter
        ]

    if max_examples:
        examples = examples[:max_examples]

    print(f"Evaluating {len(examples)} examples...")

    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        # Extract example data
        prompt = example["prompt"]
        prompt_lines = prompt.split("\n")
        bm25_query = "\n".join(prompt_lines[-10:])
        # start_index = -1
        # for i in range(len(prompt_lines) - 1, -1, -1):
        #     line = prompt_lines[i].strip()
        #     if line.startswith("def ") or line.startswith("class "):
        #         start_index = i
        #         break

        # if start_index != -1:
        #     # If a definition is found, use the context from its start
        #     retrieval_lines = prompt_lines[start_index:]
        # else:
        #     # Fallback to the last 50 lines if no 'def' or 'class' is found
        #     retrieval_lines = prompt_lines[-100:]
        groundtruth = example["groundtruth"]
        metadata = example["metadata"]
        query = query_format.format(
            repo_name=repo_path,
            file_name=metadata["file"].replace(".py", "").replace("/", "."),
            code=prompt,
        )
        # Check if query exceeds 7000 tokens
        if len(GLOBAL_TOKENIZER.encode(query)) > 3000:
            # Split prompt into lines
            lines = prompt.split("\n")

            # Get all import lines
            import_lines = [line for line in lines if "import" in line]
            import_text = "\n".join(import_lines)

            # Calculate remaining token budget
            base_query = query_format.format(
                repo_name=repo_path,
                file_name=metadata["file"].replace(".py", "").replace("/", "."),
                code="",
            )
            base_tokens = len(GLOBAL_TOKENIZER.encode(base_query))
            import_tokens = len(GLOBAL_TOKENIZER.encode(import_text))
            remaining_tokens = 3000 - base_tokens - import_tokens

            # Add bottom lines until we hit the token limit
            bottom_lines = []
            for line in reversed(lines):
                if "import" not in line:  # Skip import lines since we already have them
                    test_bottom = "\n".join(reversed(bottom_lines + [line]))
                    if len(GLOBAL_TOKENIZER.encode(test_bottom)) < remaining_tokens:
                        bottom_lines.append(line)
                    else:
                        break

            # Combine imports + bottom lines
            bottom_text = "\n".join(reversed(bottom_lines))
            truncated_code = import_text + "\n...\n" + bottom_text

            # Recreate query with truncated code
            query = query_format.format(
                repo_name=repo_path,
                file_name=metadata["file"].replace(".py", "").replace("/", "."),
                code=truncated_code,
            )

        # Get retrieved results based on retriever type
        retrieved_results = []
        if retriever_type in ["graph", "hybrid"]:
            print("Retrieving graph results")
            graph_results = stage_1(graph, query, model, tokenizer)
            if graph_results and isinstance(graph_results, list):
                retrieved_results = graph_results
                print("Successfully retrieved graph results")
            else:
                retrieved_results = []
        # Get BM25 results if needed
        bm25_retrieved_results = []
        if retriever_type in ["bm25", "hybrid"] and bm25_searcher:
            print("Retrieving BM25 results")
            current_file = metadata["file"]  # Get current file path
            bm25_results = bm25_searcher.search_excluding_file(
                bm25_query, exclude_file=current_file, top_k=100
            )
            bm25_retrieved_results = [result["text"] for result in bm25_results]
            print(f"Successfully retrieved {len(bm25_retrieved_results)} BM25 results")
        # Format retrieved context and prepend to prompt (with token limit)
        prompt_tokens = len(GLOBAL_TOKENIZER.encode(prompt))
        max_context_tokens = min(
            CONFIG["processing"]["max_context_tokens"],
            CONFIG["processing"]["max_model_len"] - prompt_tokens,
        )
        enhanced_prompt = format_retrieved_context(
            retrieved_results,
            bm25_retrieved_results,
            GLOBAL_TOKENIZER,
            prompt,
            max_context_tokens=max_context_tokens,
        )

        # Generate next line using global code model
        try:
            predicted = generate_with_code_model(
                enhanced_prompt, max_new_tokens=CONFIG["processing"]["max_new_tokens"]
            )
            print("Successfully generated code")
            print("Groundtruth: ", groundtruth)
        except Exception as e:
            print(f"Error generating code for example {i}: {e}")
            predicted = ""

        # Compute metrics
        em = compute_exact_match(predicted, groundtruth)
        es = compute_edit_similarity(predicted, groundtruth)

        # Store detailed results
        result_entry = {
            "example_idx": i,
            "task_id": metadata.get("task_id", ""),
            "repository": metadata.get("repository", ""),
            "file": metadata.get("file", ""),
            "predicted": predicted,
            "groundtruth": groundtruth,
            "exact_match": em,
            "edit_similarity": es,
            "num_retrieved": len(retrieved_results),
        }

        results.append(result_entry)

        # Update counters
        total_examples += 1
        if em:
            exact_matches += 1
        edit_similarities.append(es)

        # Print progress every 20 examples
        if (i + 1) % 20 == 0:
            current_em_rate = exact_matches / total_examples
            current_avg_es = sum(edit_similarities) / len(edit_similarities)
            print(
                f"Progress: {i+1}/{len(examples)} | EM: {current_em_rate:.3f} | Avg ES: {current_avg_es:.3f}"
            )

        # except Exception as e:
        #     print(f"Error processing example {i}: {e}")
        #     # Store error information
        #     error_entry = {
        #         "example_idx": i,
        #         "task_id": metadata.get("task_id", "") if "metadata" in example else "",
        #         "repository": (
        #             metadata.get("repository", "") if "metadata" in example else ""
        #         ),
        #         "file": metadata.get("file", "") if "metadata" in example else "",
        #         "error": str(e),
        #         "exact_match": False,
        #         "edit_similarity": 0.0,
        #         "context_tokens": 0,
        #     }
        # results.append(error_entry)
        # total_examples += 1
        # edit_similarities.append(0.0)

    # Calculate final metrics
    em_rate = exact_matches / total_examples if total_examples > 0 else 0
    avg_edit_similarity = (
        sum(edit_similarities) / len(edit_similarities) if edit_similarities else 0
    )

    evaluation_summary = {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "em_rate": em_rate,
        "average_edit_similarity": avg_edit_similarity,
        "max_edit_similarity": max(edit_similarities) if edit_similarities else 0,
        "min_edit_similarity": min(edit_similarities) if edit_similarities else 0,
    }

    return {"summary": evaluation_summary, "detailed_results": results}


def save_results(results: Dict[str, Any], filename_prefix: str = "crosscode_eval"):
    """
    Save evaluation results to files.

    Args:
        results: Results dictionary from evaluate_retrieval
        filename_prefix: Prefix for output files
    """
    # Save summary as JSON
    with open(f"{filename_prefix}_summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)

    # Save detailed results as CSV
    df = pd.DataFrame(results["detailed_results"])
    df.to_csv(f"{filename_prefix}_detailed_results.csv", index=False)

    print(
        f"Results saved to {filename_prefix}_summary.json and {filename_prefix}_detailed_results.csv"
    )

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Examples: {summary['total_examples']}")
    print(f"Exact Matches: {summary['exact_matches']}")
    print(f"EM Rate: {summary['em_rate']:.3f}")
    print(f"Average Edit Similarity: {summary['average_edit_similarity']:.3f}")
    print(f"Max Edit Similarity: {summary['max_edit_similarity']:.3f}")
    print(f"Min Edit Similarity: {summary['min_edit_similarity']:.3f}")
