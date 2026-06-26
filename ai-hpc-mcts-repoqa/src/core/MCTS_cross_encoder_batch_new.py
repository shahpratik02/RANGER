import json
import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from FlagEmbedding import FlagReranker
from transformers import AutoTokenizer
from src.utils.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Represents a node in the MCTS tree"""

    graph_node_id: str
    graph_node_data: Dict[str, Any]
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    simulation_reward: float = 0.0
    simulation_visits: int = 0
    children_reward: float = 0.0
    children_visits: int = 0
    is_tree_leaf: bool = True
    has_children_in_graph: bool = False
    references_expanded: bool = False
    is_reference_expansion: bool = False
    is_fully_expanded: bool = False
    retrieval_score: float = 0.0
    cosine_similarity: float = 0.0
    redundant_visits = 0
    selected_in_iterations: List[Dict] = field(default_factory=list)

    def add_child(self, child: "TreeNode"):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        self.is_tree_leaf = False

    def get_average_reward(self) -> float:
        """Get average reward for this node"""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def get_retrieval_score(
        self, similarity_cache: Dict[str, float], alpha: float = 0.3, iteration: int = 0
    ) -> float:
        """
        Calculates a retrieval score for a node based on a combination of factors.

        Args:
            similarity_cache: A cache of node similarities to the query.
            alpha: Weight for combining the MCTS value and cosine similarity.

        Returns:
            The final retrieval score.
        """
        if self.visit_count == 0:
            return 0.0

        # 1. MCTS-derived value (average reward from all simulations through this node)
        # value_score = self.get_average_reward()
        value_score = self.simulation_reward / self.simulation_visits
        # 2. Cosine similarity to query (scaled from 0-1 to 0-10)
        # print(self.graph_node_id)
        cosine_similarity_val = similarity_cache.get(self.graph_node_id, 0.0)
        # print(cosine_similarity_val)
        self.cosine_similarity = cosine_similarity_val
        cosine_score = cosine_similarity_val * 10.0

        # 3. Combine value score and cosine similarity using beta as a weight.
        # This balances the learned MCTS value against the initial heuristic similarity.
        combined_score = alpha * value_score + (1 - alpha) * cosine_score

        # 4. Factor in visit count as a confidence/importance metric.
        # Using log1p to dampen the effect of very high visit counts.
        # visit_confidence = math.log1p(self.visit_count)
        # visit_confidence = (
        #     2 * (iteration - self.visit_count) / iteration
        #     if self.visit_count > iteration / 2
        #     else 0
        # )
        # visit_confidence = min(self.visit_count, 2*iteration) / iteration
        visit_confidence = 0.0
        final_score = combined_score + visit_confidence

        self.retrieval_score = final_score

        return final_score

    def is_leaf_in_graph(self) -> bool:
        """Check if this node is a leaf in the original graph"""
        return not self.has_children_in_graph


class RLEnhancedGraphRAG:
    """RL-Enhanced Repository-Level GraphRAG implementation using MCTS"""

    def __init__(
        self,
        graph: Neo4jGraph,
        embedding_model: SentenceTransformer,
        exploration_param: float = None,
        reward_threshold: float = None,
        reference_threshold: float = None,
        max_iterations: int = None,
        context_limit: int = None,
        alpha: float = None,
        top_k_references: int = None,
        cross_encoder_model: str = None,
        top_k_children: int = None,
        min_top_k_children: int = None,
        reduce_top_k_flag: bool = None
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        
        # Use config values as defaults
        self.exploration_param = exploration_param if exploration_param is not None else CONFIG['mcts']['exploration_param']
        self.reward_threshold = reward_threshold if reward_threshold is not None else CONFIG['mcts']['reward_threshold']
        self.reference_threshold = reference_threshold if reference_threshold is not None else CONFIG['mcts']['reference_threshold']
        self.max_iterations = max_iterations if max_iterations is not None else CONFIG['mcts']['max_iterations']
        self.context_limit = context_limit if context_limit is not None else CONFIG['mcts']['context_limit']
        self.alpha = alpha if alpha is not None else CONFIG['mcts']['alpha']
        self.top_k_references = top_k_references if top_k_references is not None else CONFIG['mcts']['top_k_references']
        self.original_top_k_children = top_k_children if top_k_children is not None else CONFIG['mcts']['top_k_children']
        self.min_top_k_children = min_top_k_children if min_top_k_children is not None else CONFIG['mcts']['min_top_k_children']
        self.reduce_top_k_flag = reduce_top_k_flag if reduce_top_k_flag is not None else CONFIG['mcts']['reduce_top_k_flag']
        
        self.current_iteration = 0
        self.root_tree_node = None
        self.nodes_in_tree = set()  # Track unique nodes in the tree
        self.similarity_cache = {}  # Cache for cosine similarity scores
        self.root_first_expansion = True
        # Simple list to store selected node in each iteration
        self.selected_nodes_per_iteration = []
        
        # Initialize cross-encoder
        cross_encoder_model = cross_encoder_model or CONFIG['models']['cross_encoder']
        logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = FlagReranker(cross_encoder_model, use_fp16=False)
        self.cross_encoder.model = self.cross_encoder.model.to("cuda")
        logger.info("Cross-encoder model loaded successfully")
        
        # Initialize tokenizer matching the cross-encoder model
        logger.info(f"Loading tokenizer for cross-encoder model: {cross_encoder_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        logger.info("Tokenizer loaded successfully")

    def search(self, query: str, repo_name: str = "Repository") -> List[Dict]:
        """
        Main MCTS algorithm for RL-Enhanced Repository-Level GraphRAG

        Args:
            query: User query string
            repo_name: Name of the repository to search in

        Returns:
            List of high_reward_nodes
        """
        logger.info(
            f"[MCTS] Starting search for query: '{query}' in repo: '{repo_name}'"
        )
        self.nodes_in_tree = set()  # Track unique nodes in the tree
        self.similarity_cache = {}  # Clear previous similarity cache
        self.root_first_expansion = True
        self.current_iteration = 0  # ← ADD THIS LINE
        self.root_tree_node = None  # ← ADD THIS LINE
        self.selected_nodes_per_iteration = []  # Clear for new search
        self.top_k_children = self.original_top_k_children
        # Initialize MCTS tree with root node (repository)
        root_graph_node = self._get_repo_root_node(repo_name)
        if not root_graph_node:
            logger.error(f"[MCTS] Repository '{repo_name}' not found")
            return []

        root_tree_node = self._create_tree_node(root_graph_node)
        self.root_tree_node = root_tree_node
        self.root_tree_node.simulation_visits = 1
        query_embedding = self._create_embedding(query)
        logger.info(f"Query embedding created!")
        logger.info(
            f"[MCTS] Initialized tree with root: {root_tree_node.graph_node_id}"
        )
        logger.info(
            f"[MCTS] Starting {self.max_iterations} iterations with reward_threshold={self.reward_threshold}"
        )

        # Get total nodes in graph for this repository
        total_graph_nodes_query = f"""
        MATCH (r:Repo {{name:'{repo_name}'}})-[:CONTAINS]-(mod:Module)-[:CONTAINS]-(c:Class)
        OPTIONAL MATCH (c)-[:HAS_METHOD]-(m)
        RETURN count(mod)+count(c)+count(m)+1 as total_nodes
        """
        result = self.graph.query(total_graph_nodes_query)
        total_graph_nodes = result[0]['total_nodes'] if result else 0
        logger.info(f"[MCTS] Total graph nodes: {total_graph_nodes}")

        # MCTS main loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            logger.info(
                f"[MCTS] ===== Iteration {iteration + 1}/{self.max_iterations} ====="
            )
            
            # Check if entire graph is explored
            if len(self.nodes_in_tree) >= total_graph_nodes:
                logger.info(f"[MCTS] Entire graph explored ({len(self.nodes_in_tree)}/{total_graph_nodes} nodes), stopping early")
                break

            # Phase 1: Selection
            selected_node = self._select_tree_node(
                root_tree_node, query_embedding=query_embedding
            )

            # Store iteration selection info
            selection_info = {
                'iteration': iteration,
                'node_id': selected_node.graph_node_id,
                'name': selected_node.graph_node_data.get('name', 'Unknown'),
                'type': selected_node.graph_node_data.get('node_type', 'Unknown'),
                'visit_count': selected_node.visit_count,
                'total_reward': selected_node.total_reward,
                'avg_reward': selected_node.get_average_reward() if selected_node.visit_count > 0 else 0.0,
                'sim_reward': selected_node.simulation_reward / selected_node.simulation_visits if selected_node.simulation_visits > 0 else 0.0,
                'simulation_visits': selected_node.simulation_visits
            }
            selected_node.selected_in_iterations.append(selection_info)
            # Simply store the selected node info for this iteration
            self.selected_nodes_per_iteration.append(selection_info)

            logger.info(
                f"[MCTS] Phase 1 - Selected: {selected_node.graph_node_id} (visits: {selected_node.visit_count}, avg_reward: {selected_node.get_average_reward():.2f})"
            )

            # Phase 2: Correlation Expansion
            expanded_nodes = []
            if selected_node.has_children_in_graph:
                logger.info(
                    f"[MCTS] Phase 2 - Expanding tree leaf: {selected_node.graph_node_id}"
                )
                expanded_nodes = self._correlation_expansion(
                    selected_node, query_embedding
                )
                if expanded_nodes:
                    logger.info(
                        f"[MCTS] Phase 2 - Expanded to {len(expanded_nodes)} nodes: {[node.graph_node_id for node in expanded_nodes]}"
                    )
                else:
                    selected_node.is_fully_expanded = True
                    selected_node.redundant_visits += 1
                    selected_node.visit_count += 1
                    logger.info(f"[MCTS] Phase 2 - No expansion possible")
            else:
                selected_node.is_fully_expanded = True
                selected_node.redundant_visits += 1
                selected_node.visit_count += 1
                logger.info(
                    f"[MCTS] Phase 2 - No expansion needed (is_leaf: {selected_node.is_tree_leaf}, has_children: {selected_node.has_children_in_graph})"
                )

            # Phase 3: Batch Simulation and Evaluation
            nodes_with_rewards = []

            if expanded_nodes:
                logger.info(
                    f"[MCTS] Phase 3 - Batch simulating {len(expanded_nodes)} correlation nodes"
                )
                correlation_rewards, unnormalized_correlation_rewards = (
                    self._batch_evaluate_with_cross_encoder(expanded_nodes, query)
                )

                # Combine nodes with their rewards
                for node, reward, unnormalized_reward in zip(
                    expanded_nodes,
                    correlation_rewards,
                    unnormalized_correlation_rewards,
                ):
                    nodes_with_rewards.append((node, reward))
                    logger.info(
                        f"[MCTS] Phase 3 - {node.graph_node_id}: {reward:.2f} (unnormalized: {unnormalized_reward:.2f})"
                    )

            # Phase 4: Reference Expansion for high-reward nodes
            # COMMENTED OUT - Reference expansion disabled
            # reference_nodes = []
            # for node, reward in nodes_with_rewards:
            #     if reward >= self.reward_threshold and not node.references_expanded:
            #         logger.info(
            #             f"[MCTS] Phase 4 - Expanding references for: {node.graph_node_id} (reward: {reward:.2f} ≥ {self.reward_threshold})"
            #         )
            #         ref_nodes = self._expand_reference_relationships(
            #             node, query_embedding
            #         )
            #         reference_nodes.extend(ref_nodes)
            #         node.references_expanded = True

            # Phase 5: Batch Evaluation of Reference Nodes
            # COMMENTED OUT - Reference expansion disabled
            # if reference_nodes:
            #     logger.info(
            #         f"[MCTS] Phase 5 - Batch simulating {len(reference_nodes)} reference nodes"
            #     )
            #     reference_rewards, unnormalized_reference_rewards = (
            #         self._batch_evaluate_with_cross_encoder(reference_nodes, query)
            #     )

            #     # Add reference nodes with their rewards
            #     for node, reward, unnormalized_reward in zip(
            #         reference_nodes, reference_rewards, unnormalized_reference_rewards
            #     ):
            #         nodes_with_rewards.append((node, reward))
            #         logger.info(
            #             f"[MCTS] Phase 5 - {node.graph_node_id}: {reward:.2f} (unnormalized: {unnormalized_reward:.2f})"
            #         )

            # Phase 6: Batch Backpropagation
            if nodes_with_rewards:
                logger.info(
                    f"[MCTS] Phase 4 - Batch backpropagating {len(nodes_with_rewards)} nodes"
                )
                self._batch_backpropagate(nodes_with_rewards)
            else:
                logger.info(f"[MCTS] Phase 4 - No nodes to backpropagate")

        # Extract high-reward nodes
        high_reward_nodes = self._extract_high_reward_nodes(root_tree_node)

        logger.info(f"[MCTS] ===== Search Complete =====")
        logger.info(f"[MCTS] Total high-reward nodes : {len(high_reward_nodes)}")
        logger.info(
            f"[MCTS] Root node final stats - visits: {root_tree_node.visit_count}, avg_reward: {root_tree_node.get_average_reward():.2f}"
        )
        return high_reward_nodes

    def _select_tree_node(
        self,
        root: TreeNode,
        exploration_param: float = None,
        query_embedding: np.ndarray = None,
    ) -> TreeNode:
        """
        Phase 1: Node Selection using UCT (Upper Confidence bound applied to Trees)
        """
        if exploration_param is None:
            exploration_param = self.exploration_param

        current = root
        flag = False
        while not current.is_tree_leaf:

            best_child = None
            best_uct = -float("inf")

            # Select best child using UCT formula
            # Collect all unvisited children first
            unvisited_children = []
            visited_children = []

            for child in current.children:
                if child.visit_count == 0:
                    unvisited_children.append(child)
                else:
                    visited_children.append(child)

            if unvisited_children:
                if len(unvisited_children) == 1:
                    best_child = unvisited_children[0]
                    logger.info(
                        f"[MCTS] Phase 1 - Selected single unvisited child: {best_child.graph_node_id}"
                    )
                else:
                    # Multiple unvisited children - select by cosine similarity
                    best_similarity = -1.0
                    for child in unvisited_children:
                        similarity = self._get_cached_similarity(
                            query_embedding, child.graph_node_data
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_child = child

                    logger.info(
                        f"[MCTS] Phase 1 - Selected best unvisited child by similarity: {best_child.graph_node_id} (similarity: {best_similarity:.3f})"
                    )
                return best_child
            else:
                # No unvisited children - use UCT formula on visited children
                for child in visited_children:
                    # UCT formula: w_i/n_i + c*sqrt(2*ln(n_p)/n_i)
                    exploitation = child.total_reward / child.visit_count
                    exploration = exploration_param * math.sqrt(
                        2 * math.log(current.visit_count) / child.visit_count
                    )
                    uct_value = exploitation + exploration

                    if child.is_fully_expanded:
                        uct_value *= 0.0

                    if uct_value > best_uct:
                        best_uct = uct_value
                        best_child = child

            current = best_child

        # If we reached a graph leaf that's been visited multiple times,
        # find best expandable node in entire MCTS tree
        if (
            current.is_leaf_in_graph() or current.is_fully_expanded
        ) and current.visit_count >= 2:
            # logger.info(
            #     f"[MCTS] Graph leaf over-visited: {current.graph_node_id}, searching entire tree"
            # )
            # # flag = True
            # best_expandable = self._find_best_expandable_node(root, exploration_param)
            # if best_expandable and best_expandable != current:
            #     logger.info(
            #         f"[MCTS] Found better expandable node: {best_expandable.graph_node_id}"
            #     )
            #     return best_expandable
            logger.info(
                f"[MCTS] Graph leaf over-visited: {current.graph_node_id}, traversing up the tree"
            )
            # flag = True
            while current is not None and current.is_fully_expanded and all(
                child.is_fully_expanded for child in current.children
            ):
                current = current.parent
            if current is None:
                logger.info("[MCTS] Reached root/None during traversal, returning root")
                return root
            if not current.is_fully_expanded:
                logger.info(f"[MCTS] Found unexpanded node: {current.graph_node_id}")
                return current
            else:
                for child in current.children:
                    uct = -float("inf")
                    if not child.is_fully_expanded:
                        exploitation = child.total_reward / child.visit_count
                        exploration = exploration_param * math.sqrt(
                            2 * math.log(current.visit_count) / child.visit_count
                        )
                        uct_value = exploitation + exploration
                        if uct_value > uct:
                            uct = uct_value
                            best_child = child
                logger.info(
                    f"[MCTS] Found best unexpanded child: {best_child.graph_node_id}"
                )
                return best_child

        return current

    def _can_expand_node(self, node: TreeNode) -> bool:
        """
        Check if a node can be expanded (has children in original graph not yet in MCTS tree)
        """
        if node.is_leaf_in_graph():
            return False

        children_in_graph = self._get_children_from_graph(node.graph_node_data)
        existing_child_ids = {child.graph_node_id for child in node.children}

        # Check if there are children in graph not yet in MCTS tree
        for child_data in children_in_graph:
            child_id = self._get_node_identifier(child_data)
            if (
                child_id not in existing_child_ids
                and child_id not in self.nodes_in_tree
            ):
                return True

        return False

    def _correlation_expansion(
        self, node: TreeNode, query_embedding: np.ndarray
    ) -> List[TreeNode]:
        """
        Phase 2: Correlation-based Node Expansion
        Returns all expanded nodes for batch processing
        """
        children_in_graph = self._get_children_from_graph(node.graph_node_data)
        if not children_in_graph:
            logger.info(
                f"[MCTS] Phase 2 - No children found in graph for: {node.graph_node_id}"
            )
            node.is_fully_expanded = True
            return []

        logger.info(
            f"[MCTS] Phase 2 - Found {len(children_in_graph)} children, evaluating similarity"
        )

        # Calculate similarity scores for all children and sort by score (descending)
        scored_children = []
        for child_data in children_in_graph:
            similarity_score = self._get_cached_similarity(query_embedding, child_data)
            if similarity_score > 0.0:  # Only include nodes with valid embeddings
                scored_children.append((similarity_score, child_data))

        if not scored_children:
            logger.info(f"[MCTS] Phase 2 - No children with embeddings found")
            return []

        # Collect non-duplicate children with their similarities
        valid_children = []
        duplicates_skipped = 0

        for similarity_score, child_data in scored_children:
            # Check if this would be a duplicate
            node_id = self._get_node_identifier(child_data)
            if node_id not in self.nodes_in_tree:
                valid_children.append((similarity_score, child_data))
            else:
                duplicates_skipped += 1
                logger.debug(
                    f"[MCTS] Phase 2 - Skipping duplicate child (similarity: {similarity_score:.3f})"
                )

        if not valid_children:
            # All children were duplicates
            node.is_fully_expanded = True
            logger.info(
                f"[MCTS] Phase 2 - All {len(scored_children)} children were duplicates"
            )
            return []

        # Sort by similarity score (highest first)
        valid_children.sort(key=lambda x: x[0], reverse=True)

        # Add top 3 children (or fewer if less than 3 available)
        if self.reduce_top_k_flag:
            # Check if this is the root node's first expansion
            if node == self.root_tree_node and self.root_first_expansion:
                # First expansion at root - use original top_k_children
                self.root_first_expansion = False
                logger.info(f"[MCTS] Phase 2 - First root expansion, using top_k_children: {self.top_k_children}")
            else:
                # Subsequent expansions - reduce by 2 but keep minimum
                self.top_k_children = max(self.min_top_k_children, self.top_k_children//2)
                logger.info(f"[MCTS] Phase 2 - Subsequent expansion, reduced top_k_children: {self.top_k_children}")
        top_children_count = min(self.top_k_children, len(valid_children))
        added_children = []

        for i in range(top_children_count):
            similarity_score, child_data = valid_children[i]
            new_tree_node = self._create_tree_node(child_data)
            if new_tree_node:  # Should always succeed since we filtered duplicates
                node.add_child(new_tree_node)
                logger.info(f"Added child: {new_tree_node.graph_node_id}")
                added_children.append(new_tree_node)
            else:
                # This shouldn't happen since we pre-filtered duplicates, but just in case
                logger.error(f"[MCTS] Phase 2 - Unexpected duplicate after filtering")

        logger.info(
            f"[MCTS] Phase 2 - Added {len(added_children)} children for batch processing"
        )
        return added_children

    def _count_tokens(self, text: str) -> int:
        """Count exact tokens in text"""
        return len(self.tokenizer.encode(text))

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks with max_tokens each"""
        if self._count_tokens(text) <= max_tokens:
            return [text]
        
        # Split into halves recursively
        mid = len(text) // 2
        left_part = text[:mid]
        right_part = text[mid:]
        
        chunks = []
        chunks.extend(self._split_text_by_tokens(left_part, max_tokens))
        chunks.extend(self._split_text_by_tokens(right_part, max_tokens))
        
        return chunks

    def extract_purpose(self, description: str) -> str:
        """
        Extracts the PURPOSE section from a structured description.
        If not found, returns the first paragraph instead.
        """
        match = re.search(
            r"\*\*Purpose\*\*:?[\n\s]*(.+?)(?=\n\s*\*\*|$)",
            description,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        else:
            # Extract the first paragraph (non-empty block before first double line break or '**')
            paragraphs = re.split(r"\n\s*\n", description.strip())
            for para in paragraphs:
                stripped_para = para.strip()
                if stripped_para and not stripped_para.lower().startswith(
                    "**key features**"
                ):
                    return stripped_para
            return "Purpose not found."

    # Old single-node simulation function removed - replaced with batch processing

    def _batch_evaluate_with_cross_encoder(
        self, nodes: List[TreeNode], query: str
    ) -> List[float]:
        """
        Batch Cross-encoder based Node Evaluation with token-based chunking

        Args:
            nodes: List of TreeNode objects to evaluate
            query: User query

        Returns:
            List of relevance scores between 0 and 10
        """
        if not nodes:
            return []

        try:
            all_pairs = []
            node_chunk_counts = []  # Track how many chunks each node has
            
            for node in nodes:
                node_type = self._get_node_type(node.graph_node_data)
                node_data = node.graph_node_data
                
                if node_type == "Module":
                    # Keep processing as is
                    description = node_data.get("description", "")
                    all_pairs.append((query, description))
                    node_chunk_counts.append(1)
                    
                elif node_type == "Class":
                    # Check if code tokens > 14000
                    code = node_data.get("code", "")
                    code_tokens = self._count_tokens(code)
                    
                    if code_tokens > 14000:
                        # Large class - treat like Module
                        description = node_data.get("description", "")
                        all_pairs.append((query, description))
                        node_chunk_counts.append(1)
                    else:
                        # Process like other nodes
                        description = node_data.get("description", "")
                        member_descriptions = node_data.get("member_descriptions", "")
                        
                        # Check total tokens
                        combined_text = description + member_descriptions
                        query_tokens = self._count_tokens(query)
                        combined_tokens = self._count_tokens(combined_text)
                        
                        if query_tokens + combined_tokens < 8192:
                            # Can process normally
                            final_description = f"**DESCRIPTION** {description}\n**MEMBERS** {member_descriptions}"
                            all_pairs.append((query, final_description))
                            node_chunk_counts.append(1)
                        else:
                            # Need to split
                            chunks = self._split_text_by_tokens(combined_text, 8192 - query_tokens)
                            for chunk in chunks:
                                all_pairs.append((query, chunk))
                            node_chunk_counts.append(len(chunks))
                            
                else:
                    # Others (function/methods/globalvariables)
                    description = node_data.get("description", "")  # Take entire description
                    member_descriptions = node_data.get("member_descriptions", "")
                    
                    # Check total tokens
                    combined_text = description + member_descriptions
                    query_tokens = self._count_tokens(query)
                    combined_tokens = self._count_tokens(combined_text)
                    
                    if query_tokens + combined_tokens < 8192:
                        # Can process normally
                        final_description = f"{description}\n{member_descriptions}" if member_descriptions else description
                        all_pairs.append((query, final_description))
                        node_chunk_counts.append(1)
                    else:
                        # Need to split
                        chunks = self._split_text_by_tokens(combined_text, 8192 - query_tokens)
                        for chunk in chunks:
                            all_pairs.append((query, chunk))
                        node_chunk_counts.append(len(chunks))

            # Get scores for all pairs
            if all_pairs:
                scores = self.cross_encoder.compute_score(all_pairs, normalize=True)
                unnormalized_scores = self.cross_encoder.compute_score(all_pairs, normalize=False)
                
                # Ensure scores is a list
                if isinstance(scores, (int, float)):
                    scores = [scores]
                    unnormalized_scores = [unnormalized_scores]
                elif isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                    unnormalized_scores = unnormalized_scores.tolist()
            else:
                scores = []
                unnormalized_scores = []

            # Aggregate scores for nodes with multiple chunks (take max)
            final_scores = []
            final_unnormalized_scores = []
            score_idx = 0
            
            for chunk_count in node_chunk_counts:
                if chunk_count == 1:
                    final_score = scores[score_idx] * 10.0
                    final_unnormalized = unnormalized_scores[score_idx]
                    score_idx += 1
                else:
                    # Take max score from chunks
                    chunk_scores = scores[score_idx:score_idx + chunk_count]
                    chunk_unnormalized = unnormalized_scores[score_idx:score_idx + chunk_count]
                    final_score = max(chunk_scores) * 10.0
                    final_unnormalized = max(chunk_unnormalized)
                    score_idx += chunk_count
                
                final_scores.append(max(0.0, min(10.0, final_score)))  # Clamp between 0 and 10
                final_unnormalized_scores.append(final_unnormalized)

            logger.info(
                f"[MCTS] Batch evaluated {len(nodes)} nodes, scores: {[f'{s:.2f}' for s in final_scores]}"
            )
            return final_scores, final_unnormalized_scores

        except Exception as e:
            logger.warning(f"Batch cross-encoder evaluation failed: {e}")
            return [4.0] * len(nodes), [4.0] * len(nodes)  # Return default scores for all nodes

    def _batch_backpropagate(self, nodes_with_rewards: List[Tuple[TreeNode, float]]):
        """
        Batch Backpropagation for multiple nodes and their rewards
        """
        total_nodes_updated = 0

        for node, reward in nodes_with_rewards:
            current = node
            nodes_updated = 0
            is_first_node = True

            while current is not None:
                # Update visit count and total reward
                current.visit_count += 1
                current.total_reward += reward

                # Track simulation vs children rewards
                if is_first_node:
                    # This is the node that was directly simulated/evaluated
                    current.simulation_reward += reward
                    current.simulation_visits += 1
                    is_first_node = False
                else:
                    # This is a parent node receiving reward from children
                    current.children_reward += reward
                    current.children_visits += 1

                nodes_updated += 1
                current = current.parent

            total_nodes_updated += nodes_updated
            logger.info(
                f"[MCTS] Backpropagated reward {reward:.2f} for {node.graph_node_id}, updated {nodes_updated} nodes in path"
            )

        logger.info(
            f"[MCTS] Batch backpropagation complete - Updated {total_nodes_updated} total node visits"
        )

    # COMMENTED OUT - Reference expansion disabled
    # def _expand_reference_relationships(
    #     self, node: TreeNode, query_embedding: np.ndarray
    # ) -> List[TreeNode]:
    #     """
    #     Reference Relationship Expansion
    #     Returns all expanded reference nodes for batch processing
    #     """
    #     graph_node = node.graph_node_data
    #     added_tree_nodes = []

    #     # Get called nodes and caller nodes
    #     called_nodes = self._get_called_nodes(graph_node)
    #     caller_nodes = self._get_caller_nodes(graph_node)
    #     reference_nodes = called_nodes + caller_nodes

    #     scored_references = []
    #     for ref_node in reference_nodes:
    #         similarity = self._get_cached_similarity(query_embedding, ref_node)
    #         if similarity > self.reference_threshold:
    #             scored_references.append((similarity, ref_node))

    #     # Sort by similarity (highest first) and take top k
    #     scored_references.sort(key=lambda x: x[0], reverse=True)
    #     top_references = scored_references[: self.top_k_references]

    #     for similarity, ref_node in top_references:
    #         if not self._exists_in_children(node, ref_node):
    #             ref_tree_node = self._create_tree_node(ref_node)
    #             if ref_tree_node:
    #                 ref_tree_node.is_reference_expansion = True
    #                 node.add_child(ref_tree_node)
    #                 added_tree_nodes.append(ref_tree_node)
    #             else:
    #                 logger.info(
    #                     f"[MCTS] Phase 4 - Skipping duplicate reference child: {ref_node.get('name', 'unknown')}"
    #                 )

    #     if added_tree_nodes:
    #         node_details = ", ".join(
    #             [
    #                 f"{tree_node.graph_node_data.get('name', 'unknown')} (sim: {self._get_cached_similarity(query_embedding, tree_node.graph_node_data):.3f})"
    #                 for tree_node in added_tree_nodes
    #             ]
    #         )
    #         logger.info(
    #             f"[MCTS] Phase 4 - Added top {len(added_tree_nodes)} reference children to: {node.graph_node_id} - {node_details}"
    #         )
    #     else:
    #         logger.info(
    #             f"[MCTS] Phase 4 - Added {len(added_tree_nodes)} reference children to: {node.graph_node_id}"
    #         )

    #     return added_tree_nodes

    def _extract_high_reward_nodes(self, root: TreeNode) -> List[Dict]:
        """
        Extract High-Reward Nodes using BFS traversal, sorted by avg_reward descending
        and limited by context size
        """
        high_reward_nodes = []
        queue = [root]

        # Collect all high-reward nodes first
        while queue:
            node = queue.pop(0)

            if node.visit_count > 0:
                avg_reward = node.get_retrieval_score(
                    self.similarity_cache,
                    alpha=self.alpha,
                    iteration=self.max_iterations,
                )
                node_info = {
                    "node_data": node.graph_node_data,
                    "avg_reward": avg_reward,
                    "visit_count": node.visit_count,
                    "is_reference_expansion": node.is_reference_expansion,
                    "tree_node":node
                }
                high_reward_nodes.append(node_info)

            queue.extend(node.children)

        # Sort by average reward in descending order
        high_reward_nodes.sort(key=lambda x: x["avg_reward"], reverse=True)

        # Separate Module nodes from other nodes
        module_nodes = []
        other_nodes = []

        for node_info in high_reward_nodes:
            node_data = node_info["node_data"]
            if (
                node_data.get("node_type") == "Module"
                or node_data.get("node_type") == "Repo"
            ):
                module_nodes.append(node_info)
            else:
                other_nodes.append(node_info)

        # Add nodes within context limit (excluding Module nodes from context calculation)
        final_nodes = []
        current_context_size = 0

        # First add all Module nodes
        # final_nodes.extend(module_nodes)

        # Then add other nodes until context limit is reached
        count=0
        for node_info in other_nodes:
            node_data = node_info["node_data"]
            code = node_data.get("code", "")
            code_size = len(code)
            logger.info(
                f"node_id: {node_data.get('graph_node_id', 'unknown')}, code_size: {code_size}, current_context_size: {current_context_size}"
            )
            # if current_context_size + code_size <= self.context_limit:
            #     final_nodes.append(node_info)
            #     current_context_size += code_size
            # else:
            #     break
            if count < 10:
                final_nodes.append(node_info)
                count += 1
            else:
                break

        return final_nodes

    def _get_repo_root_node(self, repo_name: str) -> Optional[Dict]:
        """Get the repository root node from Neo4j"""
        query = "MATCH (r:Repo {name: $repo_name}) RETURN r"
        try:
            result = self.graph.query(query, {"repo_name": repo_name})
            if result:
                node_data = dict(result[0]["r"])
                node_data["node_type"] = "Repo"  # Ensure node_type is set
                return node_data
        except Exception as e:
            logger.error(f"Failed to get repo root node: {e}")
        return None

    def _create_tree_node(self, graph_node_data: Dict) -> TreeNode:
        """Create a tree node from graph node data"""
        node_id = self._get_node_identifier(graph_node_data)

        if node_id in self.nodes_in_tree:
            logger.debug(f"[MCTS] Node already exists in tree: {node_id}")
            return None

        has_children = self._check_has_children_in_graph(graph_node_data)
        self.nodes_in_tree.add(node_id)
        return TreeNode(
            graph_node_id=node_id,
            graph_node_data=graph_node_data,
            has_children_in_graph=has_children,
        )

    def _create_temp_node(self, graph_node_data: Dict) -> TreeNode:
        """Create a temporary tree node for simulation"""
        # return self._create_tree_node(graph_node_data)
        node_id = self._get_node_identifier(graph_node_data)
        has_children = self._check_has_children_in_graph(graph_node_data)

        return TreeNode(
            graph_node_id=node_id,
            graph_node_data=graph_node_data,
            has_children_in_graph=has_children,
        )

    def _get_node_identifier(self, node_data: Dict) -> str:
        """Generate a unique identifier for a node based on database constraints"""
        node_type = node_data.get("node_type", "Unknown")

        if node_type == "Module":
            # Constraint: m.name IS UNIQUE
            return f"Module:{node_data.get('name', 'Unknown')}"

        elif node_type == "Class":
            # Constraint: (c.name, c.signature, c.module_name) IS UNIQUE
            name = node_data.get("name", "Unknown")
            signature = node_data.get("signature", "")
            module_name = node_data.get("module_name", "Unknown")
            return f"Class:{name}:{signature}:{module_name}"

        elif node_type == "Function":
            # Constraint: (f.name, f.signature, f.module_name) IS UNIQUE
            name = node_data.get("name", "Unknown")
            signature = node_data.get("signature", "")
            module_name = node_data.get("module_name", "Unknown")
            return f"Function:{name}:{signature}:{module_name}"

        elif node_type == "Method":
            # Constraint: (m.name, m.class, m.signature, m.module_name) IS UNIQUE
            name = node_data.get("name", "Unknown")
            class_name = node_data.get("class", "Unknown")
            signature = node_data.get("signature", "")
            module_name = node_data.get("module_name", "Unknown")
            return f"Method:{name}:{class_name}:{signature}:{module_name}"

        elif node_type == "GlobalVariable":
            # Constraint: (g.name, g.module_name) IS UNIQUE
            name = node_data.get("name", "Unknown")
            module_name = node_data.get("module_name", "Unknown")
            return f"GlobalVariable:{name}:{module_name}"

        elif node_type == "Repo":
            # Repo identifier
            name = node_data.get("name", "Repository")
            return f"Repo:{name}"

        else:
            # Fallback for unknown types
            return f"{node_type}:{str(hash(str(node_data)))}"

    def _check_has_children_in_graph(self, node_data: Dict) -> bool:
        """Check if a node has children in the graph"""
        node_type = node_data.get("node_type", "")

        if node_type == "Repo":
            query = "MATCH (r:Repo {name: $name})-[:CONTAINS]->(child) RETURN count(child) as count"
            params = {"name": node_data.get("name")}

        elif node_type == "Module":
            query = "MATCH (m:Module {name: $name})-[:CONTAINS]->(child) RETURN count(child) as count"
            params = {"name": node_data.get("name")}

        elif node_type == "Class":
            query = """
            MATCH (c:Class {name: $name, signature: $signature, module_name: $module_name})
            -[:HAS_METHOD|HAS_FIELD]->(child) 
            RETURN count(child) as count
            """
            params = {
                "name": node_data.get("name"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "Function":
            # Functions can have children via USES relationships
            query = """
            MATCH (f:Function {name: $name, signature: $signature, module_name: $module_name})
            -[:USES]->(child) 
            RETURN count(child) as count
            """
            params = {
                "name": node_data.get("name"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "Method":
            # Methods can have children via USES relationships
            query = """
            MATCH (m:Method {name: $name, class: $class, signature: $signature, module_name: $module_name})
            -[:USES]->(child) 
            RETURN count(child) as count
            """
            params = {
                "name": node_data.get("name"),
                "class": node_data.get("class"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "GlobalVariable":
            # GlobalVariables can have children via USES relationships
            query = """
            MATCH (g:GlobalVariable {name: $name, module_name: $module_name})
            -[:USES]->(child) 
            RETURN count(child) as count
            """
            params = {
                "name": node_data.get("name"),
                "module_name": node_data.get("module_name"),
            }

        else:
            return False

        try:
            result = self.graph.query(query, params)
            return result[0]["count"] > 0 if result else False
        except Exception:
            return False

    def _get_children_from_graph(self, node_data: Dict) -> List[Dict]:
        """Get children of a node from the graph"""
        node_type = node_data.get("node_type", "")
        children = []

        if node_type == "Repo":
            query = "MATCH (r:Repo {name: $name})-[:CONTAINS]->(child) RETURN child, labels(child) as node_type"
            params = {"name": node_data.get("name")}

        elif node_type == "Module":
            query = "MATCH (m:Module {name: $name})-[:CONTAINS|USES]->(child) RETURN child, labels(child) as node_type"
            params = {"name": node_data.get("name")}

        elif node_type == "Class":
            query = """
            MATCH (c:Class {name: $name, signature: $signature, module_name: $module_name})
            -[:HAS_METHOD|HAS_FIELD|USES|INHERITS]-(child) 
            RETURN child, labels(child) as node_type
            """
            params = {
                "name": node_data.get("name"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "Function":
            # Functions have children via USES relationships
            query = """
            MATCH (f:Function {name: $name, signature: $signature, module_name: $module_name})
            -[:USES]-(child) 
            RETURN child, labels(child) as node_type
            """
            params = {
                "name": node_data.get("name"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "Method":
            # Methods have children via USES relationships
            query = """
            MATCH (m:Method {name: $name, class: $class, signature: $signature, module_name: $module_name})
            -[:USES]-(child) 
            RETURN child, labels(child) as node_type
            """
            params = {
                "name": node_data.get("name"),
                "class": node_data.get("class"),
                "signature": node_data.get("signature", ""),
                "module_name": node_data.get("module_name"),
            }

        elif node_type == "GlobalVariable":
            # GlobalVariables have children via USES relationships
            query = """
            MATCH (g:GlobalVariable {name: $name, module_name: $module_name})
            -[:USES]-(child) 
            RETURN child, labels(child) as node_type
            """
            params = {
                "name": node_data.get("name"),
                "module_name": node_data.get("module_name"),
            }

        else:
            return children

        try:
            result = self.graph.query(query, params)
            for record in result:
                child_data = dict(record["child"])
                child_data["node_type"] = (
                    record["node_type"][0] if record["node_type"] else "Unknown"
                )
                children.append(child_data)
        except Exception as e:
            logger.warning(f"Failed to get children: {e}")

        return children

    # COMMENTED OUT - Reference expansion disabled
    # def _get_called_nodes(self, node_data: Dict) -> List[Dict]:
    #     """Get nodes that this node calls (USES relationship)"""
    #     node_name = node_data.get("name")
    #     node_type = node_data.get("node_type", "")

    #     if node_type in ["Function", "Method", "GlobalVariable"]:
    #         query = f"""
    #         MATCH (n:{node_type} {{name: $name}})-[:USES]->(called)
    #         RETURN called, labels(called) as node_type
    #         """
    #         params = {"name": node_name}

    #         try:
    #             result = self.graph.query(query, params)
    #             called_nodes = []
    #             for record in result:
    #                 called_data = dict(record["called"])
    #                 called_data["node_type"] = (
    #                     record["node_type"][0] if record["node_type"] else "Unknown"
    #                 )
    #                 called_nodes.append(called_data)
    #             return called_nodes
    #         except Exception as e:
    #             logger.warning(f"Failed to get called nodes: {e}")

    #     return []

    # COMMENTED OUT - Reference expansion disabled
    # def _get_caller_nodes(self, node_data: Dict) -> List[Dict]:
    #     """Get nodes that call this node (reverse USES relationship)"""
    #     node_name = node_data.get("name")
    #     node_type = node_data.get("node_type", "")

    #     if node_type in ["Function", "Method", "Class", "GlobalVariable"]:
    #         query = f"""
    #         MATCH (caller)-[:USES]->(n:{node_type} {{name: $name}})
    #         RETURN caller, labels(caller) as node_type
    #         """
    #         params = {"name": node_name}

    #         try:
    #             result = self.graph.query(query, params)
    #             caller_nodes = []
    #             for record in result:
    #                 caller_data = dict(record["caller"])
    #                 caller_data["node_type"] = (
    #                     record["node_type"][0] if record["node_type"] else "Unknown"
    #                 )
    #                 caller_nodes.append(caller_data)
    #             return caller_nodes
    #         except Exception as e:
    #             logger.warning(f"Failed to get caller nodes: {e}")

    #     return []

    # COMMENTED OUT - Reference expansion disabled
    # def _exists_in_children(self, node: TreeNode, ref_node_data: Dict) -> bool:
    #     """Check if a reference node already exists in children"""
    #     ref_id = self._get_node_identifier(ref_node_data)
    #     for child in node.children:
    #         if child.graph_node_id == ref_id:
    #             return True
    #     return False

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text using the embedding model"""
        return self.embedding_model.encode(text)

    def _cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            return cosine_similarity(emb1, emb2)[0][0]
        except Exception:
            return 0.0

    def _get_cached_similarity(
        self, query_embedding: np.ndarray, node_data: Dict
    ) -> float:
        """Get cosine similarity with memoization"""
        node_id = self._get_node_identifier(node_data)

        if node_id in self.similarity_cache:
            return self.similarity_cache[node_id]

        if "embedding" in node_data and node_data["embedding"]:
            node_embedding = np.array(node_data["embedding"])
            similarity = self._cosine_similarity(query_embedding, node_embedding)
        else:
            logger.info(f"No embedding found for node: {node_id}")
            similarity = 0.0

        self.similarity_cache[node_id] = similarity
        return similarity

    def _extract_code_content(self, node_data: Dict) -> str:
        """Extract code content from node data"""
        return node_data.get("code", "")

    def _get_node_type(self, node_data: Dict) -> str:
        """Get node type from node data"""
        return node_data.get("node_type", "Unknown")

    def count_tree_nodes(self, root: TreeNode) -> int:
        """Count total nodes in MCTS tree"""
        count = 0
        queue = [root]

        while queue:
            node = queue.pop(0)
            count += 1
            queue.extend(node.children)

        return count


def main(query: str):
    """Example usage of RL-Enhanced GraphRAG with batch cross-encoder processing"""

    # Configuration
    graph_config = CONFIG['neo4j']

    # Initialize components
    graph = Neo4jGraph(**graph_config)
    embedding_model = SentenceTransformer(CONFIG['models']['embedding'])

    # Initialize RL-Enhanced GraphRAG with batch cross-encoder processing
    # Will use all config defaults
    rl_graphrag = RLEnhancedGraphRAG(
        graph=graph,
        embedding_model=embedding_model
    )

    # Perform search
    high_reward_nodes = rl_graphrag.search(query)

    # Display results
    print(f"\nFound {len(high_reward_nodes)} high-reward nodes:")
    for i, node_info in enumerate(high_reward_nodes, 1):
        print(
            f"{i}. {node_info['node_data'].get('name', 'Unknown')} "
            f"(Reward: {node_info['avg_reward']:.2f}, "
            f"Visits: {node_info['visit_count']})"
        )

    return [x["node_data"]["code"] for x in high_reward_nodes]


if __name__ == "__main__":
    query = "Tell me about the agent module and its capabilities."

    main(query)
