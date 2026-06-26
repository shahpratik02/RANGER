59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 

1 

2 

3 

4 

5 

6 

7 

10 

11 12 13 14 

15 16 17 18 19 

33 

34 

35 

36 

37 

38 

39 

40 

41 

42 

43 

44 

45 

46 

47 

48 

49 50 

56 

## **RANGER: Repository-level Agent for Graph-Enhanced Retrieval** 

Anonymous Author(s) 

## **Abstract** 

7 General-purpose automated software engineering (ASE) includes 8 tasks such as code completion, retrieval, repair, QA, and summariza9 tion. These tasks require a code retrieval system that can handle spe10 cific queries about code entities, or _code entity queries_ (for example, 11 locating a specific class or retrieving the dependencies of a function), 12 as well as general queries without explicit code entities, or _natural_ 13 _language queries_ (for example, describing a task and retrieving the 14 corresponding code). We present **RANGER** , a repository-level code 15 retrieval agent designed to address both query types, filling a gap 16 in recent works that have focused primarily on code-entity queries. 17 We first present a tool that constructs a comprehensive knowledge 18 graph of the entire repository, capturing hierarchical and cross19 file dependencies down to the variable level, and augments graph 20 nodes with textual descriptions and embeddings to bridge the gap 21 between code and natural language. RANGER then operates on this 22 graph through a dual-stage retrieval pipeline. Entity-based queries 23 are answered through fast Cypher lookups, while natural language 24 queries are handled by MCTS-guided graph exploration. We evalu25 ate RANGER across four diverse benchmarks that represent core 26 ASE tasks including code search, question answering, cross-file 27 dependency retrieval, and repository-level code completion. On 28 CodeSearchNet and RepoQA it outperforms retrieval baselines that 29 use embeddings from strong models such as Qwen3-8B. On Re30 poBench, it achieves superior cross-file dependency retrieval over 31 baselines, and on CrossCodeEval, pairing RANGER with BM25 de32 livers the highest exact match rate in code completion compared to other RAG methods. 

## **CCS Concepts** 

• **Computing methodologies** → **Artificial Intelligence** ; • **Information systems** → **Information retrieval** . 

## **Keywords** 

Repository-Level Code Retrieval, Code Knowledge Graph, Monte Carlo Tree Search, Graph Traversal 

## **ACM Reference Format:** 

Anonymous Author(s). 2026. RANGER: Repository-level Agent for GraphEnhanced Retrieval. In _Proceedings of Agentic Software Engineering (SE 3.0) @ KDD (Agentic SE ’26)._ ACM, New York, NY, USA, 10 pages. https: //doi.org/XXXXXXX.XXXXXXX 

50 Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed 51 for profit or commercial advantage and that copies bear this notice and the full citation 52 on the first page. Copyrights for components of this work owned by others than the 53 author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, orrepublish, to post on servers or to redistribute to lists, requires prior specific permission 54 and/or a fee. Request permissions from permissions@acm.org. 55 _Agentic SE ’26, Jeju, Korea_ 

© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2025/07 https://doi.org/XXXXXXX.XXXXXXX 

## **1 Introduction** 

Retrieving relevant code snippets, functions, and classes from large repositories is central to modern software engineering, as the quality of retrieved context underpins downstream tasks for AI agents and large language models, including code generation, patch generation, automated program repair, and repository-level code completion. Our focus is on repository-level agent workflows such as analysis, patching, and planning, where retrieval is one step in a longer loop that may include multiple model calls and tool or test execution, and where precise retrieval is crucial for correctness. While retrieval over natural language has seen rapid progress [20, 21], code retrieval remains substantially more challenging. Unlike natural language, code often contains long-range and multi-hop dependencies [1], where the semantics of a program may depend on variables, function calls, or imports that appear far apart in the source. These properties render simple flat indexing insufficient for code retrieval, motivating the use of graph databases [28] and multi-hop reasoning to capture cross-file relationships, call graphs, and dependency chains [15, 39]. 

An additional challenge in code retrieval arises from query diversity. _Code-entity queries_ ask questions about specific code-entities (e.g., "What are the dependencies of Calculator class?"). In contrast, _natural language queries_ , describe behaviors or constraints without naming symbols (e.g., "Where do we implement addition?"). Natural language queries [9, 31] are particularly difficult due to the semantic gap between natural and symbolic languages [17–19, 48], as well as embedding anisotropy and hubness in code representations [22, 49]. 

Graph retrieval offers a promising direction by enabling multihop traversal while preserving hierarchical relationships, in contrast to flat index RAG [51]. By modeling the repository as a graph, where nodes correspond to code entities and edges encode hierarchical or dependency links, GraphRAG can resolve queries that require following transitive dependencies, such as tracing a function call across multiple intermediate layers or modules. However, current graph-based code retrieval methods tend to perform well on code-entity or structure-aware queries, but lack dedicated support for open-ended natural language queries [24, 27, 37]. 

To address these challenges, we develop an efficient knowledge graph construction procedure together with a Monte Carlo Tree Search (MCTS)-based graph traversal algorithm. Using an agentic architecture, we integrate the knowledge graph with MCTS to enable a dual-stage retrieval system capable of handling both symbolic code-entity queries and natural language queries. Our key contributions are as follows: 

- **Efficient Knowledge Graph Construction for Code Retrieval:** A tool to transform Python repositories into an information-rich knowledge graph that captures hierarchical and cross-file dependencies by parsing abstract syntax trees (AST). To mitigate the semantic gap between natural and symbolic coding languages, we augment graph 

114 

57 

115 

116 

58 

1 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

Anon. 

175 

176 

177 

178 

179 

180 

181 

182 

183 

184 

185 

186 

187 

188 

189 

190 

191 

192 193 

194 

195 

196 

197 

198 

199 

200 

201 

202 203 204 205 

144 

145 

146 

147 

206 207 208 209 210 211 212 213 214 215 216 217 218 219 

148 

149 

150 

151 

152 153 

154 

155 

157 

158 159 

160 

161 

162 163 164 165 166 

220 

221 

222 223 224 225 226 227 

228 229 230 231 

- 117 nodes with textual descriptions of code entities and their 118 corresponding embeddings. 119 • **Monte Carlo Tree Search-Based Graph Traversal Al-** 120 **gorithm:** A graph traversal algorithm inspired by Monte 121 Carlo Tree Search that balances exploration and exploita122 tion. Starting from a source node, it quickly expands to 123 promising candidates using a bi-encoder. During the simu124 lation phase, a cross-encoder computes reward scores for 125 visited nodes. Over time, rollouts uncover the most relevant 126 node for retrieval. 127 • **Router Retrieval Agent:** A dual-stage retrieval pipeline 128 that routes queries by type. Code-entity queries are resolved 129 through fast Cypher lookups on the graph database, while 130 natural language queries fall back to the MCTS-based graph 131 traversal algorithm. 

132 

## 133 **2 Related Work** 

134 

Early neural models for source code established that structure135 aware encoders using Abstract Syntax Tree (AST) paths (e.g., code2vec 136 [4], code2seq [3]) or graph neural networks [34] [2] could outper137 form lexical approaches. Subsequently, Transformer-based pretrain138 ing became the dominant paradigm, with models like Codex [6], 139 CodeGen [36], CodeLlama [42], StarCoder2 [30], and DeepSeek140 Coder [16] demonstrating strong performance on function- and 141 file-level tasks. However, these models condition on local context 142 and struggle to incorporate the cross-file dependencies essential 143 for reasoning in large repositories. 

Early retrieval-augmented generation (RAG) systems such as RECODE [56], REDCODER [39], and TreeGen [46] injected external code snippets into prompts. These methods treated code as flat text, relying on lexical or vector similarity, which hindered their ability to reason across multiple files. While later work improved recall, it remained snippet-centric and failed to model the typed, multi-hop relationships that connect definitions and uses across a codebase. 

_Natural Language Code Search._ Natural language–based code search has been extensively studied, beginning with large-scale benchmarks such as CodeSearchNet [19], which enabled systematic evaluation of neural retrieval models. Subsequent work enriched code embeddings with structural signals, including program dependency graphs 58, [53] and variable flow graphs ([55], 55), while efficiency-focused methods like ExCS [18] improved scalability through offline code expansion. More recently, repository-level approaches employ multi-stage pipelines that integrate commit metadata with BERT re-rankers [14] or translate natural language queries into domain-specific query languages [23]. In parallel, query reformulation [40] and LLM-driven paraphrasing[33] highlight the central challenge of aligning vague natural descriptions with precise code identifiers, especially in large and evolving repositories. 

156 

> 167 _Graph-Based Retrieval and Agentic Frameworks._ Graph-centric 

> 168 methods address structural limitations by explicitly encoding re- 

> 169 lationships like definitions, references, and calls, but they differ 

> 170 significantly in scope, persistence, and query support. Some ap- 

> 171 proaches build local graphs, for instance, GraphCoder [27] creates 

> 172 Code Context Graphs for snippets but omits cross-file links. Cat- 

> 173 Coder [38] constructs on-the-fly type-dependency subgraphs for 

statically-typed languages, sacrificing the persistent, long-range relationships needed at repository scale. 

Repository-scale graphs improve coverage but introduce tradeoffs. RepoGraph [37] separates definitions and references into distinct nodes with basic invoke/contain edges, which creates redundancy and lacks semantic embeddings for text-code alignment. CoCoMIC [12] models cross-file relations at the file level through imports rather than direct function-to-function edges, constraining multi-hop precision. RepoFuse [24] uses Jedi-based analysis to build an in-memory graph of imports, inheritance, and calls but focuses on rule-based neighbor capture for completion. Similarly, DraCo [8] constructs a fine-grained, variable-level dataflow graph with typed edges (Assigns, Refers, Typeof) but remains specialized for code completion tasks. CodeGraphModel [47] integrates a repository graph into an LLM via a graph-adapter but relies on lightweight analysis and a simple retrieval method based on entity extraction and string matching, limiting its support for non-entity and multi-hop queries. 

A growing line of work couples LLMs with code graphs in agentic frameworks. LocAgent [7] converts entire codebases into directed graphs and exposes tools like SearchEntity and TraverseGraph, but its comprehensive traversals can be computationally expensive without a persistent graph database. OrcaLoca [54] uses prioritybased scheduling and in-memory NetworkX graphs derived from ASTs but acknowledges that its incomplete reference analysis can miss semantic dependencies. CodexGraph [28] bridges LLM agents with graph databases for structure-aware retrieval, but its workflows often rely on explicit identifiers, making purely natural language queries challenging. MCTS-based agents like LingmaAgent [32] explore code graphs with LLM-based reward estimation, while related variants such as RTSoG [29] and REKG-MCTS [45] apply similar strategies to document and text knowledge graphs, but the repeated high-fidelity LLM scoring incurs significant inference cost and can introduce nondeterminism. These trends highlight a need for agents that combine persistent, semantically augmented graphs with cost-aware planning to balance accuracy and efficiency. 

This work presents **RANGER** , a repository-level retrieval agent that integrates persistent graph construction with query-type– aware retrieval. A repository-wide knowledge graph is built through AST parsing and enriched with semantic descriptions and embeddings. At query time, RANGER first converts the input into a Cypher query over this graph. For _code-entity queries_ , these Cypher lookups typically suffice for direct resolution. For _natural language queries_ , which often fail to return direct matches, RANGER invokes an MCTS-based graph exploration that combines bi-encoder expansion with selective cross-encoder scoring. This dual-path design enables efficient handling of both symbolic and natural language queries, overcoming the limitations of flat embedding indices and gaps of prior graph-based retrieval methods. 

## **3 Methodology** 

## **3.1 Overall Architecture** 

We propose a retrieval agent capable of processing both _natural language_ and _code-entity_ queries for code retrieval. As mentioned earlier, natural language queries are challenging due to the semantic gap between textual descriptions and code embeddings [13, 19]. 

174 

232 

2 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

RANGER: Repository-level Agent for Graph-Enhanced Retrieval 

291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 

267 268 269 270 271 272 273 274 275 276 277 278 

279 

280 

281 282 283 284 

285 286 287 288 

233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 

257 258 259 260 

261 

262 

263 264 265 266 

**Figure 1: Overview of our proposed system. The offline phase constructs a knowledge graph from the code repository with entity descriptions and embeddings. The online phase handles two query types: code-entity queries are resolved via Cypher lookups, while natural language queries trigger MCTS-based graph traversal.** 

As illustrated in Figure 1, the system uses a two-stage pipeline with an _offline indexing_ stage for repository preprocessing and graph construction and an _online query_ stage for retrieval and reasoning with RANGER. In the offline stage, a code repository is parsed into an entity graph stored in a graph database (e.g., Neo4j). This includes AST parsing to build the knowledge graph, LLMassisted description generation for components and modules, and embedding computation for those descriptions. 

Cypher first and invoking MCTS only when Cypher returns no meaningful results. This avoids adding a separate learned router that could introduce additional latency and a new failure mode. The following subsections detail the components of this architecture. 

## **3.2 Code Parsing and Knowledge Graph Creation** 

The repository-level knowledge graph is constructed through a two-stage process that first builds isolated file-level graphs and then stitches them into a unified repository-level graph. This design ensures that intra-file structures are captured accurately before resolving complex inter-file dependencies. Figure 3.2 provides an illustrative two-file example of the construction and stitching procedure. 

In the online stage, RANGER first converts the user query into a Cypher statement via zero-shot LLM prompting (examples in Table 1). For _code-entity queries_ , these Cypher lookups typically suffice for direct response generation (Path 1). For _natural language queries_ , the input often contains no explicit repository symbol to anchor the lookup, so the LLM may generate an underspecified or brittle Cypher query. As illustrated in Table 1, the query “How to get database table name” does not name a module/class/function, so the model may guess an entity anchor and produce a constraint such as (m:Module {name: ’database’}). If the repository does not contain a module with that exact name or uses a different naming convention, the lookup returns None despite relevant code existing elsewhere. In these cases, we fall back to MCTS (Path 2). We intentionally keep this routing logic simple by always attempting 

_Stage 1: File-level parsing._ Each file is processed using the tree-sitter library [5], which produces a detailed Abstract Syntax Tree (AST). This contrasts with existing systems [24, 28] that rely on Pythonspecific tools like Jedi or Parso. We traverse the AST to extract key code entities and relationships, which are organized into an intermediate JSON object serving as a decoupled transfer representation. A database-specific ingestion component then converts 

289 

290 

3 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

Anon. 

400 

401 

402 

403 

**==> picture [560 x 630] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||||||||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|349|Input|Cypher|Schema element|Attributes|407|
|350351|Natural language query:|database table name”|“How to get|MATCH|(r:Repo) -[:CONTAINS]->(|Repo node|name.|408409|
|352|m:Module|{name:|'|Module node|name, local_name, embedding,description .|410|
|database '})|Class node|name, signature, code, module_name, embedding,|
|353|-[:CONTAINS]->(c:Class)|411|
|354|RETURN|c.name ,|c.code|description, member_descriptions.|412|
|Function node|name, signature, code, module_name, embedding,|
|355|413|
|Code-entity query|description, member_descriptions.|
|356|“Fetch the most important dependencies|MATCH|(m:Module|{name:|'tests|Method node|name, signature, code, module_name, class,|414|
|357|from the repository to complete the|.test_renderables '})|415|
|following code”:|-[:CONTAINS]->(f:|embedding, description, member_descriptions.|
|358|Function|{name:|'|Field node|name, code, class, embedding, description,|416|
|359|Input:|file_name:|test_renderables|member_descriptions.|417|
|tests.test_renderables|'})|
|360|Incomplete code:|OPTIONAL|MATCH|(f) -[:USES]->(|GlobalVariable|name, code, module_name, embedding, description,|418|
|361|@reference ()|dep)|node|member_descriptions.|419|
|362|def|test_renderables(viewer:|RETURN|DISTINCT|dep.name|AS|Import node|name, module, alias, dotted_folder_name.|420|
|363|grid_xzViewer):=|np.mgrid|name ,signature ,dep.signaturedep.code|ASAS|CONTAINS edge|Repo→Module; Module→{Module, Class, Function,|421|
|364|[ -1.5:1.5:0.3 ,|code|GlobalVariable}.|422|
|365|-1.5:1.5:0.3]|423|
|366|...|HAS_METHOD edge|Class→Method.|424|
|line_strip|=|np.zeros ((2|*|HAS_FIELD edge|Class→Field.|
|367368|line_strip [::2]n_lines ,|3))=|line_starts|INHERITS edge|Class→Class (base class).|425426|
|line_strip [1::2]|=|line_ends|USES edge|{Class, Function, Method,|
|369|427|
|GlobalVariable}→{Class, Function, Method,|
|370|Table|1:|LLM-generated|Cypher|examples|for|two|query|GlobalVariable}.|428|
|371|types.|429|
|372|Table 2: RANGER knowledge-graph schema.|430|
|373|431|
|374|432|
|375376|these|objects|into|nodes|and|edges|in|the|graph|database.|This|Files|inside|a|repository|———————>|Create|File|Level|Graph|————>|Merge|Import|Nodes|433434|
|separation allows new programming languages to be supported|7|#|base.py|
|377|32|precision|=|2|(>)|435|
|by|modifying|only|the|AST|parser,|and|new|graph|backends|by|4|def|format_resutt():|‘|
|378379380|updatingthe|RANGERonlyknowledgethe|ingestiongraph,module.includingThe|completethe|sevenontologynode|typesof|rT)11268|classdefreturnCateulator:multiply(self,return"roundeda+b|toa,{digits}b):|places"|@apy(aa)AS|&eo|.|-|436437438|
|and five relationship types, is detailed in Table 2. Unlike existing|3B|return|a+b|.|
|381|439|
|382383|approaches [RANGER maintains a fine-grained representation down to classfields and global variables. Within each file, structural edges are32, 47] that often operate at the file or function level,|a472|#fromdefextended.pyquickbase|add():import|Calculator,|precision,|format_result|©|@\oed|p=_|fSe@oyoOH|440441|
|384|immediately established, including CONTAINS edges from a Module|9|def|divide(self,|a,|b):|a|1.|-.|»|442|
|385|to|its|classes|and|functions,|HAS_METHOD|edges|from|a|Class|to|n102|placesreturn|=round(aprecision/|b,|places)|oO|‘[~-] aE|Ld~|443|
|386|its|methods,|and|INHERITS|edges|to|represent|class|inheritance.|131“15|def|deno():sciprint(sei.multiply(3,=|Scientific()|4))|.a|@Ld|444|
|387|To handle unresolved dependencies, temporary Import nodes are|uv16|print|format_resutt())|445|
|388|created, pointing to entities outside the current file. Unlike exist-|446|
|389|ing approaches such as the Code Graph Model [47], which applies|447|
|390|lightweight semantic analysis, or OrcaLoca [54], which omits static|Figure 2:|Two-stage graph construction example.|448|
|391|analysis, this step explicitly preserves placeholders for cross-file|449|
|392|references.|450|
|393|We illustrate this procedure with a simple two-file repository in|451|
|394|Stage 2: Repository-level consolidation.|After all files are parsed,|Figure 3.2. In Stage 1, base.py yields nodes for the base module,|452|
|395|the|system|resolves|the|temporary|Import|nodes.|Each|Import|the|Calculator|class|and|its|methods,|a|helper|function,|and|a|453|
|396|node|is|matched|to|its|corresponding|entity|(Class,|Function,|precision global variable, connected by hierarchical edges such as|454|
|397|Module, etc.) elsewhere in the repository, and all incoming edges are|CONTAINS and HAS_METHOD. In parallel, extended.py yields nodes|455|
|398|redirected to the resolved node. This “stitching” step ensures that|for|the|extended|module|and|its|entities,|but|any|references|to|456|
|399|cross-file dependencies are explicitly represented, yielding broader|symbols defined in base.py (e.g., imported classes or globals) are|457|
|400|coverage than prior approaches such as the lightweight cross-file|represented as temporary Import placeholders. In Stage 2, we re-|458|
|401|analyses in the Code Graph Model [47] or the limited function-call|solve these Import nodes by matching them to their true targets|459|
|402|tracking in Lingma Agent [32]. Once redirected, redundant Import|elsewhere in the repository and redirecting incoming edges, e.g.,|460|
|403|nodes are deleted. The result is a repository-level knowledge graph|converting a placeholder inheritance link into a concrete INHERITS|461|
|404|that completely represents both intra-file structure and inter-file|edge|to|Calculator,|or|connecting|a|placeholder|reference|via|462|
|405|dependencies.|USES.|Finally,|the|Import|nodes|are|deleted,|leaving|a|compact|463|
|406|464|

**----- End of picture text -----**<br>


404 

405 

406 

4 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

RANGER: Repository-level Agent for Graph-Enhanced Retrieval 

523 

524 525 526 527 

528 

529 

530 531 532 533 534 535 

478 479 480 481 482 483 484 

536 

537 

538 539 540 541 

542 

543 544 545 546 

488 

547 548 

549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 

504 

505 

506 

507 

508 

509 

510 

511 

512 

513 

514 515 

516 

517 

518 519 520 521 

465 466 467 468 469 470 471 472 473 474 475 476 477 

**Figure 3: MCTS-based graph traversal.** 

485 486 repository-level graph that supports direct Cypher lookups as well 487 as multi-hop traversal. 

## 489 **3.3 LLM-Assisted Semantic Description and** 490 **Embedding** 

> 491 After constructing the knowledge graph, we add semantic attributes 

> 492 by generating natural language descriptions for each code entity 

> 493 with an LLM using a hierarchical bottom up procedure. Following 

> 494 Code2JSON [44], each entity receives two descriptions, a high level 

> 495 purpose summary and a granular member level summary of impor- 

> 496 tant variables and behaviors. For small entities such as functions 

> 497 and methods, whose source code fits within the context limit of the 

> 498 LLM, we generate both descriptions directly from code, while for 

> 499 larger entities such as modules and large classes we compose them 

> 500 from precomputed member summaries. We then concatenate the 

> 501 two descriptions, encode them into a vector embedding, and store 

> 502 the text and embedding as node attributes. Prompt templates will 

> 503 be released with the code upon acceptance. 

## **3.4 MCTS-Based Graph Traversal Algorithm** 

To efficiently search the code knowledge graph, we use Monte Carlo Tree Search (MCTS) to balance retrieval efficiency and accuracy. A bi-encoder guides exploration and a cross-encoder scores only the most promising candidates, which focuses computation where expected relevance is highest [52]. The process, formalized in Algorithm 1, consists of Selection, Expansion, Simulation, Backpropagation, and a final Extraction stage. 

**Selection.** The selection phase balances exploration (searching new parts of the graph) with exploitation (focusing on paths that have previously yielded high rewards). Starting from the root of the search tree, we recursively select the child node with the highest Upper Confidence bound for Trees (UCT) score, defined as: 

UCT( _𝑣_ ) = max _𝑅_ (1 _𝑣, 𝑁𝑣_ )[+] _[ 𝑐]_ √︄ 2 ln maxmax(1( _, 𝑁_ 1 _, 𝑁_ parent _𝑣_ ) ( _𝑣_ ) ) where _𝑅𝑣_ is the 

total reward of a node _𝑣_ , _𝑁𝑣_ is its visit count, and _𝑐_ is an exploration parameter. We continue until a leaf is reached. If that leaf is fully expanded, we backtrack to the nearest ancestor with unexpanded neighbors. 

**Expansion.** Once a leaf node is selected, the search tree is expanded by adding its neighbors from the code graph as child nodes. To guide this expansion, the bi encoder ranks all neighbors based on the cosine similarity of their embeddings with the query embedding. The top- _𝑘_ most similar and previously unvisited neighbors are then added to the search tree. This bi-encoder driven expansion serves as a fast and effective heuristic for candidate generation. 

**Simulation.** This stage evaluates the relevance of newly expanded nodes. Unlike MCTS in adversarial games [43], where rollouts simulate sequences of actions to a terminal state, our retrieval task lacks a discrete win/loss outcome. A random traversal from a node is ill-suited for determining its relevance to a query. Therefore, we redefine the simulation step as a direct relevance evaluation using a cross-encoder. The query and the node’s semantic content are used as input to the cross-encoder, which produces a precise relevance score. This score serves directly as the reward for the node. To maximize throughput, evaluations are processed in batches. 

**Backpropagation.** After evaluation we propagate the reward up the tree. For every node on the path to the root we increment its visit count ( _𝑁𝑣_ ) and add the reward to its total ( _𝑅𝑣_ ). This update guides subsequent selection toward promising regions of the code graph. 

**Extraction** After a predefined number of iterations the search terminates and we extract a ranked list of relevant code nodes. The final score for each visited node is _𝑠_ ( _𝑣_ ) = _𝛼_ · max _𝑅_ (1 _𝑣, 𝑁𝑣_ )[+][(][1][ −] 

_𝛼_ ) · sim( _𝐸𝑞, 𝐸𝑣_ ) which balances the learned MCTS reward with the initial bi encoder similarity to yield a robust final ranking. 

Figure 3 illustrates the dynamics of this procedure on the twofile example repository from Figure 1 for the natural-language query “Where is the code for addition?” In early iterations, the selection and expansion steps add high-level structural nodes such as the base module and its immediate children (e.g., precision and Calculator) based on embedding similarity. The simulation step then assigns rewards via the cross-encoder, and backpropagation increases the UCT values along successful paths. Subsequent iterations expand deeper neighbors, eventually reaching fine-grained nodes such as the add method and a quick_add helper function, which receive high rewards and are selected in the final extraction and ranking stage. 

Algorithm 1 summarizes the complete MCTS-based retrieval procedure, including the bi-encoder expansion heuristic, cross-encoder reward computation, and the final ranking score. 

## **4 Experiments** 

We evaluate RANGER on four diverse datasets spanning both _codeentity_ and _natural-language_ query types and three practical scenarios covering repository-level code retrieval, code completion, and question answering. 

522 

5 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

Anon. 

639 

640 Cypher generator (RepoBench, Meta-Llama-3.1-70B-Instruct (AWQ 641 CrossCodeEval) INT4) 642 Text description generator deepseek-coder-1.3B-instruct (CodeSearchNet, RepoQA) 643 Query embedding model (MCTS) mxbai-embed-large-v1 644 (CodeSearchNet, RepoQA) 645 Cross-encoder (MCTS reward) bge-reranker-v2-m3 646 (CodeSearchNet, RepoQA) 647 MCTS iterations _𝑇_ (CodeSearchNet 200 / 500 648 / RepoQA) 649 MCTS _𝑘_ min 20 650 MCTS _𝑘_ init | _𝑀_ |/2 (| _𝑀_ |=#Module nodes) 651 MCTS _𝛼_ (CodeSearchNet / RepoQA) 0.5 / 0.9 652 **Table 3: Key experimental settings. We report the main mod-** 653 **els and MCTS hyperparameters used across datasets.** 654 655 656 657 658 659 Graph construction scaling 660 661 20 662 663 664 15 665 666 10 667 668 669 5 670 Repos 671 Binned median (n=12) 672 0 0 250 500 750 1000 1250 673 Total graph nodes 674 675 **Figure 4: Offline graph construction scaling.** 676 677 678 679 680 a Cypher statement plus a database lookup (∼1s end-to-end in a∼1s end-to-end in a1s end-to-end in a 681 typical deployment), and varies slightly with the LLM used. For 682 natural-language queries, the MCTS path is more expensive than a 683 single-pass dense retriever, especially at higher iteration budgets. 684 In our target agentic workflows, end-to-end time is typically domi685 nated by model calls and tool or test execution, and these workloads 686 also benefit from more precise retrieval, so retrieval is often not 687 primary bottleneck. Figure 5 shows average MCTS runtime 688 on CodeSearchNet and RepoQA rising with iteration count, with 689 200 iterations taking around 14.4–25.7s and 500 iterations taking 690 around 25.8–39.4s on average. Reported graph construction time 691 includes only parsing and knowledge graph creation, and excludes 692 the semantic description/embedding stage, since this cost depends 693 strongly on the LLM used for generating descriptions and the hard694 ware used. 695 696 

**Algorithm 1** MCTS-based Graph Traversal 

- 581 582 1.5em 0.8em 583 **Require:** Query _𝑞_ with emb. _𝐸𝑞_ ; graph G = (V _,_ E) with node desc. 584 _𝐷𝑢_ and emb. _𝐸𝑢_ (when exists); root _𝑟_ ; cross-encoder _𝑔𝜙_ ( _𝑞, 𝐷_ ) ∈ R; 585 _𝑘_ init _,𝑘_ min _,𝑐, 𝛼, 𝐵,𝑇_ . 586 **Defs:** _𝑝_ ( _𝑣_ ) is parent and Ch( _𝑣_ ) children in tree T; nbr( _𝑣_ ) are neigh587 bors in G; fully expanded iff nbr( _𝑣_ ) ⊆Vtree. 588 topk(· _,𝑘_ ) returns top- _𝑘_ items by score; 589 clamp( _𝑥,_ 0 _,_ 10)= min(10 _,_ max(0 _,𝑥_ )); sim( _𝑥, 𝑦_ )= cos( _𝐸𝑥 , 𝐸𝑦_ ). uct(·) is UCT (Upper Confidence bound for Trees): 

- 590 591 uct( _𝑣_ )= max _𝑅_ (1 _𝑣,𝑁𝑣_ )[+] _[ 𝑐]_ √︂ 2 ln maxmax((11 _,𝑁,𝑁𝑣𝑝_ )( _𝑣_ ) ) . 592 1: Init tree T ←{ _𝑟_ }; for _𝑣_ ∈T set _𝑁𝑣, 𝑅𝑣, 𝑅𝑣_[(] _[𝑠]_[)] _[, 𝑁] 𝑣_[(] _[𝑠]_[)] ← 0; _𝑘_ ← _𝑘_ init; 

- 593 Vtree ←{ _𝑟_ }. 

> 594 2: **for** _𝑡_ = 1 to _𝑇_ **do** 

> 595 **(A) Select** 596 3: curr ← _𝑟_ 

- 4: **while** curr has children in T and not fully expanded **do** 5: curr ← arg max _𝑢_ ∈Ch(curr) uct( _𝑢_ ) 6: **end while** 7: **if** _𝑁_ curr ≥ 2 and nbr(curr) \ Vtree = ∅ **then** 

597 

598 

599 

600 

- 8: Ascend to expandable ancestor; if none, **continue** 

- 601 9: **end if** 

- 602 **(B) Expand** 603 10: C ← nbr(curr) \ Vtree 604 11: S ←{( _𝑢,_ sim( _𝑞,𝑢_ )) : _𝑢_ ∈C _, 𝐸𝑢_ exists} 605 12: **if** S = ∅ **then** 606 13: Mark curr fully expanded; **continue** 607 14: **end if** 608 15: E ← topk(S _,𝑘_ ) (by sim) 609 16: Add E as children of curr in T; Vtree ←Vtree ∪E 610 17: _𝑘_ ← max( _𝑘_ min _,_ ⌊ _𝑘_ /2⌋) **(C) Simulate (batched cross-encoder)** 

- 611 18: For _𝑢_ ∈E, compute _𝑠𝑢_ ← clamp(10 _𝑔𝜙_ ( _𝑞, 𝐷𝑢_ ) _,_ 0 _,_ 10) 

- 612 **(D) Backprop (batched)** 613 19: **for** each _𝑢_ ∈E **do** 614 20: **for** each _𝑣_ on path _𝑢_ ⇝ _𝑟_ in T **do** 615 21: _𝑁𝑣_ ← _𝑁𝑣_ + 1; _𝑅𝑣_ ← _𝑅𝑣_ + _𝑠𝑢_ 616 22: **if** _𝑣_ = _𝑢_ **then** 617 23: _𝑅𝑣_[(] _[𝑠]_[)] ← _𝑅𝑣_[(] _[𝑠]_[)] + _𝑠𝑢_ ; _𝑁𝑣_[(] _[𝑠]_[)] ← _𝑁𝑣_[(] _[𝑠]_[)] + 1 618 24: **end if** 25: **end for** 

- 619 26: **end for** 

- 620 27: **end for** 

- 621 **Rank** 

- 622 28: Vvis ←{ _𝑣_ ∈T : _𝑁𝑣 >_ 0} 

- 623624 29: For _𝑣_ ∈Vvis: _𝑠_ ( _𝑣_ ) ← _𝛼_ max _𝑅_ (1 _𝑣_[(] _,𝑁[𝑠]_[)] _𝑣_[(] _[𝑠]_[)] )[+][(][1][ −] _[𝛼]_[)][ 10 sim][(] _[𝑞, 𝑣]_[)] 625 30: **return** topk(Vvis _, 𝐵_ ) sorted by _𝑠_ ( _𝑣_ ) 

626 

627 628 **4.1 Efficiency and graph construction cost** 

> 629 We quantify the offline cost of repository indexing by measuring 

> 630 end-to-end graph construction time across 1258 Python repositories 

> 631 from the datasets we worked with. Figure 4 shows that construction 

> 632 time increases predictably with graph size and follows an approx- 

> 633 imately linear trend, with corr = 0 _._ 84. Median indexing time is 

> 634 6.18s per repository. Because this is offline graph creation, it does 

> 635 not affect online query latency. Online compute is dominated by 

> 636 the retrieval path. The Cypher path is typically fast because it re- 

> 637 quires a single lightweight LLM call to translate the query into 

||**Setting**|**Value**|
|---|---|---|
||Cypher generator (RepoBench,<br>CrossCodeEval)<br>Text description generator<br>(CodeSearchNet, RepoQA)<br>Query embedding model (MCTS)<br>(CodeSearchNet, RepoQA)<br>Cross-encoder (MCTS reward)<br>(CodeSearchNet, RepoQA)|Meta-Llama-3.1-70B-Instruct (AWQ<br>INT4)<br>deepseek-coder-1.3B-instruct<br>mxbai-embed-large-v1<br>bge-reranker-v2-m3|
||MCTS iterations_𝑇_(CodeSearchNet<br>/ RepoQA)|200 / 500|
||MCTS_𝑘_min<br>MCTS_𝑘_init|20<br>|_𝑀_|/2 (|_𝑀_|=#Modulenodes)|
||MCTS_𝛼_(CodeSearchNet / RepoQA)|0.5 / 0.9|



**Table 3: Key experimental settings. We report the main models and MCTS hyperparameters used across datasets.** 

a Cypher statement plus a database lookup (∼1s end-to-end in a∼1s end-to-end in a1s end-to-end in a typical deployment), and varies slightly with the LLM used. For natural-language queries, the MCTS path is more expensive than a single-pass dense retriever, especially at higher iteration budgets. In our target agentic workflows, end-to-end time is typically dominated by model calls and tool or test execution, and these workloads also benefit from more precise retrieval, so retrieval is often not the primary bottleneck. Figure 5 shows average MCTS runtime on CodeSearchNet and RepoQA rising with iteration count, with 200 iterations taking around 14.4–25.7s and 500 iterations taking around 25.8–39.4s on average. Reported graph construction time includes only parsing and knowledge graph creation, and excludes the semantic description/embedding stage, since this cost depends strongly on the LLM used for generating descriptions and the hardware used. 

638 

6 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

RANGER: Repository-level Agent for Graph-Enhanced Retrieval 

755 

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 

702 703 704 705 706 

707 

708 

709 

710 

711 

712 

713 

714 

715 

716 

717 

718 

719 

720 

721 

722 

723 

724 

725 

726 

727 

728 

729 

730 

731 

732 

733 

734 

735 

736 

737 

738 

739 

740 

741 

742 

800 801 802 803 804 805 806 807 808 809 810 811 

743 

744 

745 746 747 748 749 

750 

751 

> 697 **4.2 Natural Language Query Based Retrieval** 

> 698 _4.2.1 Datasets & Setup._ **CodeSearchNet** Challenge (Python split) 

> 699 consists of 99 natural language queries with expert relevance anno- 

> 700 tations over a large corpus of Python functions [19]. We select 70 

> 701 repositories with the highest query counts, build knowledge graphs 

> 702 from corresponding commits, and prune nodes not present in the 

> 703 official corpus to align with ground truth annotations. 

**RepoQA** originally evaluates long context code understanding via the Searching Needle Function task where multiple functions are provided to an LLM as context along with a function description and the LLM must return the corresponding function. To facilitate our evaluation we modify the task so that all functions become our corpus and the function description becomes our natural language query [25].The function description includes Purpose, Input, Output, and Procedure fields, but to better reflect realistic queries, we use only the Purpose field as the natural language query. We use the Python split with ten repositories and ten descriptions per repository. 

For both datasets we generate text descriptions and embeddings as detailed in Section 3 and run the MCTS stage for retrieval. 

_4.2.2 Baselines and Results._ We compare to two vector search baselines. The first uses raw code embeddings indexed directly from corpus chunks. The second uses embeddings of LLM generated semantic descriptions. This isolates MCTS gains beyond gains from descriptive text. 

Table 4 reports NDCG@10 and Recall@10 on CodeSearchNet and RepoQA. RANGER improves both metrics over the baselines and also exceeds retrieval with Qwen-3-8B [57] embeddings which are currently top ranked on the MTEB leaderboard [35]. The improvements stem from the use of cross-encoder scoring, which provides higher accuracy than bi-encoder similarity but is too expensive to apply exhaustively. RANGER addresses this with an MCTS-guided traversal, where the bi-encoder expands promising graph paths and the cross-encoder is applied only to high-value candidates. This selective application preserves the accuracy benefits of crossencoders while keeping retrieval computationally tractable. 

Figure 5 shows that NDCG@10 and Recall@10 improve steadily with additional MCTS iterations before the rate of improvement slows. The curves exhibit clear knees that indicate the optimal iteration range for practical deployment, balancing retrieval quality with computational cost. 

## **4.3 Code-Entity Query Based Retrieval** 

_4.3.1 Dataset & Setup._ **RepoBench** [26] evaluates repository-level retrieval via RepoBench-R, where the task is selecting the most relevant cross-file snippet to support next-line prediction. We use the Python v1.1 split and restrict to repositories with at least five data points (430 repositories). The prompt provides an incomplete in-file chunk with code entities, which RANGER converts into Cypher queries to retrieve cross-file dependencies before ranking (example in Table 1). Because commit IDs were not released and repositories changed after dataset creation, we use the latest commit as of December 31 2023 and re run baselines for consistency. Since 

> 1mxbai-embed-large-v1 

||**Metric**|**RANGER**|**Code Embed.**|**Text Embed.**|**Text Embed.**|
|---|---|---|---|---|---|
|||(MCTS iter)<br>CodeT5<br>Qwen3-8B<br>**CodeSearchNet Dataset**||Qwen3-8B|mxbai1|
||NDCG@10|**0.786**(200)|0.419<br>0.725|0.701|0.664|
||Recall@10|**0.911**(200)|0.643<br>0.891<br>**RepoQA Dataset**|0.856|0.847|
||NDCG@10|**0.741**(500)|0.718<br>0.722|0.709|0.706|
||Recall@10|**0.890**(500)|0.810<br>0.850|0.810|0.810|



**Table 4: Performance comparison on CodeSearchNet and RepoQA. RANGER consistently outperforms baseline embedding models across datasets. Iteration counts are shown in parentheses. Best baseline results are bolded.** 

**==> picture [242 x 100] intentionally omitted <==**

**Figure 5: Performance metrics across MCTS iterations for natural language query datasets. Left shows CodeSearchNet NDCG@10, Recall@10, and the average MCTS runtime per iteration across repositories and queries. Right shows RepoQA NDCG@10, Recall@10, and corresponding average runtimes.** 

all queries here are code entity queries handled directly by Stage 1 we omit text descriptions which are mainly needed for Path 2 MCTS to reduce compute. 

_4.3.2 Baselines and Results._ Following RepoBench-R setup, the baseline treats import statement snippets as candidate contexts, capturing file-level linkage. Both RANGER and the baseline use the same rerankers and the same top- _𝑘_ protocol, so differences in Accuracy@5, NDCG@5, and MRR@5 can be attributed to the retrieval mechanism rather than reranking. 

Our graph agent improves Accuracy@5, NDCG@5 and MRR@5 across rerankers which shows better localization of fine grained dependencies than file level imports. Pure semantic retrieval performs poorly which supports the need for cross-file graph traversal over linear index search. See Table 5. 

||**Reranker**|**Acc@5**<br>**RANGER Baseline **|**Acc@5**<br>**RANGER Baseline **|**NDCG@5**<br> **RANGER Baseline **|**NDCG@5**<br> **RANGER Baseline **|**MRR@5**<br> **RANGER Baseline**|**MRR@5**<br> **RANGER Baseline**|
|---|---|---|---|---|---|---|---|
||UniXcoder2|**0.5446**|0_._4346|0_._4120|0_._3075|0_._3601|0_._2509|
||Qwen3-8B|**0.5471**|0_._4940|0_._4120|0_._3530|0_._3577|0_._2919|



**Table 5: Performance comparison on the RepoBench benchmark for cross-file dependency retrieval.** 

> 2microsoft/unixcoder-base (∼100M parameters). 

752 753 

754 

812 

7 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

Anon. 

871 

872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 

823 

824 

825 

826 

827 

828 

887 

829 

888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 

830 

831 

832 

833 

834 

835 

836 

837 

838 

839 

840 

841 

842 

843 

844 

845 

846 

847 

848 

849 

850 

851 

852 

853 

854 

855 

856 

857 

858 

859 

860 

861 

862 

863 

864 

865 

923 

866 

924 925 

> 813 **4.4 Code-Entity Query Based Code Completion** 

> 814 _4.4.1 Dataset & Setup._ **CrossCodeEval** [11] tests cross file code 

> 815 completion across Python, Java, TypeScript and C# using real repos- 

> 816 itories where the correct continuation depends on cross file context 

> 817 and not just the current file. We use the Python split with 471 repos- 

> 818 itories, build knowledge graphs from the dataset specified commits, 

> 819 and retrieve cross file context via RANGER. Same as Repobench, for 

> 820 each repository, a code knowledge graph is constructed from the 

> 821 target commit, which is provided in the datasets, without creating 

> 822 text descriptions. 

_4.4.2 Baselines and Results._ We compare RANGER against BM25 and several repository level retrievers. **BM25** [41] serves as a strong sparse lexical baseline by selecting top-k contexts via termfrequency scoring. **CGM MULTI** [47] constructs a one hop ego subgraph around the active file and applies graph aware attention. **RepoFuse** [24] fuses analogy contexts with rationale contexts. **RLCoder** [50] learns a retrieval policy with perplexity based rewards and a learned stopping rule. **R2C2** [10] assembles repository aware prompts by selecting candidate snippets with context conditioning. Inspired by RepoFuse, which shows that fusing analogy and rationale contexts improves code generation, we also report **RANGER** which pairs graph based cross file retrieval with BM25. We pair RANGER with BM25 only in CrossCodeEval because code completion metrics are sensitive not only to retrieving the correct cross-file context but also to matching repository-specific lexical and stylistic conventions; in that setting, RANGER provides structural and semantic context, while BM25 contributes lexical/style information. For pure retrieval tasks, RANGER alone is sufficient for repositorylevel retrieval in real code. 

Table 4.4.2 reports Exact Match (EM) and Edit Similarity (ES) across DeepSeek Coder 7B and CodeLlama 7B. **RANGER** achieves the highest EM with DeepSeek Coder 7B and CodeLlama 7B and competitive EM with StarCoder 7B while consistently outperforming BM25. EM is a stricter metric than ES, and high EM indicates that many completions exactly match the ground truth, which is desirable. Because **RANGER** balances cross-file dependency localization (from RANGER) and lexical similarity (from BM25), it can improve correctness and EM even when ES differs slightly. _[★]_ Note: CodexGraph was evaluated on CrossCodeEval Lite (1,000 random samples) with DeepSeek Coder v2, unlike other methods evaluated on the full dataset with DeepSeek Coder v1. Due to the unavailability of its source code, we could not reproduce its results in our setup. 

## **5 Conclusion** 

We introduced RANGER, a repository level agent for graph enhanced code retrieval that handles both _code entity queries_ and _natural language queries_ . This capability is largely absent from existing code retrieval methods. Our MCTS based graph exploration algorithm, most helpful for natural language queries, uses a bi-encoder for expansion and a cross encoder as the reward. On CodeSearchNet and RepoQA we surpass strong semantic retrieval systems, including Qwen-3-8B embedding baseline [57] ranked number one on MTEB Leaderboard [35], while using smaller models for embedding and reranking mxbai-embed-large-v1 with 335M parameters 

|**Method**|**DeepSeek-7B**<br>**EM**<br>**ES**|**CodeLlama-7B**<br>**EM**<br>**ES**|**StarCoder-7B**<br>**EM**<br>**ES**|
|---|---|---|---|
|**RANGER**<br>BM25<br>CGM-MULTI<br>RepoFuse<br>RLCoder<br>R2C2<br>CodexGraph_★_|**36.27**<br>70_._77<br>28_._57<br>65_._95<br>33_._88<br>71_._19<br>27_._92<br>73_._09<br>30_._28<br>**74.42**<br>32_._70<br>54_._00<br>20_._20<br>63_._14|**31.68**<br>66_._91<br>24_._87<br>62_._83<br>31.03<br>**73.90**<br>24_._80<br>71_._05<br>26_._60<br>72_._27<br>23_._60<br>42_._90|30.80<br>66.03<br>22.33<br>69.60<br>**31.00**<br>71.66<br>24.20<br>70.82<br>25.82<br>**72.11**<br>30.90<br>51.90|



**Table 6: Performance comparison on CrossCodeEval (Python).** 

and bge-reranker-v2-m3 with 568M parameters. Because cross encoders are more accurate but expensive and often infeasible to apply over the enitre repository, MCTS scores only promising nodes, keeping quality close to exhaustive reranking at lower cost. For repository level completion, where relevant code often lives in other files and is not semantically similar to the query, our graph-guided traversal retrieves the necessary context by following structural relationships rather than embedding proximity alone. 

Although RANGER shows strong retrieval performance across multiple benchmarks, several limitations remain. The use of static offline repository graphs limits applicability to dynamic or rapidly evolving codebases where dependencies change frequently. The MCTS stage, while effective for natural language queries, introduces additional inference latency and computational cost that can be a drawback in latency-sensitive interactive settings. Our intended setting is repository-level agent workflows, which place a premium on precise retrieval and where this overhead is often a smaller part of end-to-end runtime. Node scoring currently depends on cross encoder relevance estimates, which may not be the best reward signal. 

Future work will focus on adaptability, efficiency, and evaluation breadth. One direction is incremental graph maintenance that supports live repository updates with minimal recomputation. Another direction is a multi stage retrieval agent in the ReACT style that can combine symbolic Cypher queries with targeted MCTS starting from intermediate graph nodes. This can reduce rollout depth and latency. Learned reward models, including a small language model trained for relevance scoring or reinforcement learning approaches, may offer more robust signals than a fixed cross encoder. At present RANGER supports Python repositories. Since we use the tree-sitter library, which is not Python specific and supports many languages, we plan to extend the system to additional languages. Significant work is already done and actively underway to support C++ as our immediate next step. Code and resources will be released publicly upon acceptance. 

## **References** 

[1] Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, and Charles Sutton. 2018. A Survey of Machine Learning for Big Code and Naturalness. _Comput. Surveys_ 

LLM usage. Generative AI tools were used for writing assistance. The authors are fully responsible for all technical content and claims. 

867 868 869 

926 927 928 

870 

8 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

RANGER: Repository-level Agent for Graph-Enhanced Retrieval 

987 

988 989 990 991 992 993 994 995 996 997 998 999 

1000 1001 1002 

1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 

1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 

1035 1036 1037 1038 1039 1040 

1041 

1042 1043 1044 

   - 51, 4 (2018), 1–37. 

- 929 51, 4 (2018), 1–37. 930 [2] Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. 2018. Learning to Represent Programs with Graphs. In _International Conference on Learning_ 

- 931 _Representations (ICLR)_ . 932 [3] Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. 2019. code2seq: Generating 933 Sequences from Structured Representations of Code. In _International Conference on Learning Representations (ICLR)_ . 

- 934 [4] Uri Alon, Meital Zilberstein, Omer Levy, and Eran Yahav. 2019. code2vec: Learn935 ing Distributed Representations of Code. In _Proceedings of the ACM on Program-_ 936 [5] _ming Languages (POPL)_ Max Brunsfeld. 2018. Tree-sitter: An incremental parsing system for program-, Vol. 3. ACM, 1–29. 937 ming tools. https://github.com/tree-sitter/tree-sitter. Accessed: 2026-01-22. 938 [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg 

- 939 Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, 940 Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail 941 Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fo- 

- 942 tios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex 943 Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, 944 Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight,Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, 945 Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario 946 Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. _arXiv preprint arXiv:2107.03374_ 

- 947 (2021). 

- 948 [7] Zhaoling Chen, Robert Tang, Gangda Deng, Fang Wu, Jialong Wu, Zhiwei Jiang, 949 Viktor Prasanna, Arman Cohan, and Xingyao Wang. 2025. LocAgent: GraphGuided LLM Agents for Code Localization. In _Proceedings of the 63rd Annual_ 

- 950 _Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . 951 Association for Computational Linguistics, Vienna, Austria, 8697–8727. doi:10. 18653/v1/2025.acl-long.426 

- 952 [8] Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-Guided Retrieval Augmen953 tation for Repository-Level Code Completion. In _Proceedings of the 62nd Annual_ 954 _Meeting of the Association for Computational Linguistics (ACL)_ . 7957–7977. 

- [9] Matteo Ciniselli, Nathan Cooper, Luca Pascarella, Antonio Mastropaolo, Emad 

- 955 Aghajani, Denys Poshyvanyk, Massimiliano Di Penta, and Gabriele Bavota. 2022. 956 An Empirical Study on the Usage of Transformer Models for Code Completion. 957 _IEEE Transactions on Software Engineering_ 48, 12 (2022), 4818–4837. 

- [10] Ken Deng, Jiaheng Liu, He Zhu, Congnan Liu, Jingxin Li, Jiakai Wang, Peng 

- 958 Zhao, Chenchen Zhang, Yanan Wu, Xueqiao Yin, Yuanxing Zhang, Zizheng 959 Zhan, Wenbo Su, Bangyu Xiang, Tiezheng Ge, and Bo Zheng. 2024. R2C2-Coder: 960 Abilities of Code Large Language Models.Enhancing and Benchmarking Real-world Repository-level Code Completion _arXiv preprint arXiv:2406.01359_ (2024). 961 doi:10.48550/arXiv.2406.01359 

- 962 [11] Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parmin- 

- 963 der Bhatia, Dan Roth, and Bing Xiang. 2023. CrossCodeEval: A Diverse 964 and Multilingual Benchmark for Cross-File Code Completion. In _Advances_ 965 _in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track_ . https://proceedings.neurips.cc/paper_files/paper/2023/hash/ 

- 966 920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html 

- 967 [12] Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Murali Krishna Ramanathan, 968 Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. 2024. CoCoMIC:Code Completion by Jointly Modeling In-file and Cross-file Context. In _Pro-_ 969 _ceedings of the 2024 Joint International Conference on Computational Linguistics,_ 970 _Language Resources and Evaluation (LREC-COLING)_ . 3446–3458. 

- [13] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, 

- 971 Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020. CodeBERT: 972 A Pre-Trained Model for Programming and Natural Languages. In _Findings_ 973 _of the Association for Computational Linguistics: EMNLP 2020_ . Association for Computational Linguistics, 1536–1547. doi:10.18653/v1/2020.findings-emnlp.139 

- 974 [14] Siddharth Gandhi, Luyu Gao, and Jamie Callan. 2025. Repository-level Code 975 Search with Neural Retrieval Methods. _arXiv preprint arXiv:2502.07067_ (2025). 

- 976 [15] Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022.UniXcoder: Unified Cross-Modal Pre-training for Code Representation. In _Pro-_ 977 _ceedings of the 60th Annual Meeting of the Association for Computational Lin-_ 978 _guistics (Volume 1: Long Papers)_ . Association for Computational Linguistics, 7212–7225. 

- 979 [16] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guant980 ing Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 981 2024. DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence. _arXiv preprint arXiv:2401.14196_ (2024). 

- 982 [17] Dhruv Gupta, Gayathri Ganesh Lakshmy, and Yiqing Xie. 2025. SACL: Un983 derstanding and Combating Textual Bias in Code Retrieval with Semantic984 Augmented Reranking and Localization. In _Findings of the Association for Computational Linguistics: EMNLP 2025_ . 

985 

- [18] Siwei Huang, Bo Cai, Yaoxiang Yu, and Jian Luo. 2024. ExCS: Accelerating Code Search with Code Expansion. _Scientific Reports_ 14 (2024), 29166. 

- [19] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. _arXiv preprint arXiv:1909.09436_ (2019). 

- [20] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. _Transactions on Machine Learning Research_ (2022). 

- [21] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for OpenDomain Question Answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, 6769–6781. 

- [22] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimenez, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robber, Pushmeet Kohli, Oriol Vinyals, Demis Hassabis, and Koray Kavukcuoglu. 2022. Competition-Level Code Generation with AlphaCode. _Science_ 378, 6624 (2022), 1092–1097. 

- [23] Keyu Liang, Zhongxin Liu, Chao Liu, Zhiyuan Wan, David Lo, and Xiaohu Yang. 2025. Zero-Shot Cross-Domain Code Search without Fine-Tuning. In _Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE)_ . 

- [24] Ming Liang, Xiaoheng Xie, Gehao Zhang, Xunjin Zheng, Peng Di, Wei Jiang, Hongwei Chen, Chengpeng Wang, and Gang Fan. 2024. RepoFuse: RepositoryLevel Code Completion with Fused Dual Context. _arXiv preprint arXiv:2402.14323_ (2024). https://arxiv.org/abs/2402.14323 

- [25] Jiawei Liu, Jia Le Tian, Vijay Daita, Yuxiang Wei, Yifeng Ding, Yuhan Katherine Wang, Jun Yang, and Lingming Zhang. 2024. RepoQA: Evaluating Long Context Code Understanding. _arXiv preprint arXiv:2406.06025_ (2024). doi:10.48550/arXiv. 2406.06025 

- [26] Tianyang Liu, Canwen Xu, and Julian McAuley. 2024. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems. In _International Conference on Learning Representations (ICLR)_ . https://openreview.net/forum?id=pPjZIOuQuF 

- [27] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024. GraphCoder: Enhancing Repository-Level Code Completion via Coarse-to-fine Retrieval Based on Code Context Graph. In _Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE)_ . ACM. 

- [28] Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Qizhe Shieh, and Wenmeng Zhou. 2025. CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume_ 

   - _1: Long Papers)_ . Association for Computational Linguistics, Albuquerque, New Mexico, 142–160. doi:10.18653/v1/2025.naacl-long.7 

- [29] Xiao Long, Liansheng Zhuang, Chen Shen, Shaotian Yan, Yifei Li, and Shafei Wang. 2025. Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question Answering. _arXiv preprint arXiv:2505.12476_ (2025). 

- [30] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel LamyPoirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian J. McAuley, Han Hu, Torsten Scholak, Sébastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, and et al. 2024. StarCoder 2 and The Stack v2: The Next Generation. _arXiv preprint arXiv:2402.19173_ (2024). 

- [31] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ . 

- [32] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2024. How to Understand Whole Software Repository? _arXiv preprint arXiv:2406.01422_ (2024). 

- [33] Yuetian Mao, Chengcheng Wan, Yuze Jiang, and Xiaodong Gu. 2023. SelfSupervised Query Reformulation for Code Search. In _Proceedings of the 31st_ 

986 

9 

Agentic SE ’26, August 10, 2026, Jeju, Korea 

Anon. 

1103 

1104 

1105 

1106 

1107 

1108 

1109 

1110 

1111 

1112 

1113 

1114 

1115 

1116 

- 1045 _ACM Joint European Software Engineering Conference and Symposium on the_ 1046 _Foundations of Software Engineering (ESEC/FSE)_ . ACM. [34] Lili Mou, Ge Li, Lu Zhang, Tao Wang, and Zhi Jin. 2016. Convolutional Neu- 

- 1047 ral Networks over Tree Structures for Programming Language Processing. In 1048 _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 30. 1049 [35] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB: Massive Text Embedding Benchmark. In _Proceedings of the 17th Con-_ 

- 1050 _ference of the European Chapter of the Association for Computational Linguistics_ 1051 _(EACL)_ . Association for Computational Linguistics, Dubrovnik, Croatia, 2014– 2037. doi:10.18653/v1/2023.eacl-main.148 

- 1052 [36] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, 1053 Silvio Savarese, and Caiming Xiong. 2023. CodeGen: An Open Large Language 1054 Model for Code with Multi-Turn Program Synthesis. In _International Conference on Learning Representations (ICLR)_ . 

- 1055 [37] Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, 1056 Jiawei Han, Hongming Zhang, and Dong Yu. 2025. RepoGraph: Enhancing 1057 AI Software Engineering with Repository-level Code Graph. In _International Conference on Learning Representations (ICLR)_ . 

- 1058 [38] Zhiyuan Pan, Xing Hu, Xin Xia, and Xiaohu Yang. 2024. CatCoder: Enhanc1059 ing Repository-Level Code Generation with Integrated Contextual Information. 1060 [39] _arXiv preprint arXiv:2406.03283_ Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and (2024). 1061 Kai-Wei Chang. 2021. Retrieval Augmented Code Generation and Summariza1062 tion. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ . Association for Computational Linguistics, 2719–2734. 

- 1063 [40] Mohammad Masudur Rahman, Chanchal K. Roy, and David Lo. 2019. Automatic 1064 Query Reformulation for Code Search using Crowdsourced Knowledge. _Empirical_ 1065 _Software Engineering_ 24, 4 (2019), 1869–1924. [41] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame- 

- 1066 work: BM25 and Beyond. _Foundations and Trends in Information Retrieval_ 3, 4 1067 (2009), 333–389. doi:10.1561/1500000019 

_Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, Online, 6397–6407. doi:10.18653/v1/2020.emnlp-main.519 

- [53] Ying Yin, Longfei Ma, Yuqi Gong, Yucen Shi, Fazal Wahab, and Yuhai Zhao. 2024. Deep Semantics-Enhanced Neural Code Search. _Electronics_ 13, 23 (2024), 4704. 

- [54] Zhongming Yu, Hejia Zhang, Yujie Zhao, Hanxian Huang, Matrix Yao, Ke Ding, and Jishen Zhao. 2025. OrcaLoca: An LLM Agent Framework for Software Issue Localization. In _Proceedings of the 42nd International Conference on Machine Learning (ICML)_ . https://openreview.net/forum?id=LyUfPOvM6I OpenReview paper. 

- [55] Chen Zeng, Yue Yu, Shanshan Li, Xin Xia, Zhiming Wang, Mingyang Geng, Linxiao Bai, Wei Dong, and Xiangke Liao. 2023. deGraphCS: Embedding Variablebased Flow Graph for Neural Code Search. _ACM Transactions on Software Engineering and Methodology_ 32, 2 (2023), 1–27. 

- [56] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, 2471–2484. 

- [57] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. _arXiv preprint arXiv:2506.05176_ (2025). doi:10.48550/arXiv. 2506.05176 

|||||||
|---|---|---|---|---|---|
|1059||ing Repository-Level Code Generation with Integrated Contextual Information.||[57] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,|1117|
|1060|[39]|_arXiv preprint arXiv:2406.03283_(2024).<br> Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and||Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.<br>2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through|1118|
|1061||Kai-Wei Chang. 2021. Retrieval Augmented Code Generation and Summariza-||Foundation Models. _arXiv preprint arXiv:2506.05176_(2025). doi:10.48550/arXiv.|1119|
|1062||tion. In_Findings of the Association for Computational Linguistics: EMNLP 2021_.||2506.05176|1120|
|1063|[40]|Association for Computational Linguistics, 2719–2734.<br> Mohammad Masudur Rahman, Chanchal K. Roy, and David Lo. 2019. Automatic||[58] Yue Zou, Bihuan Ban, Yinxing Xue, and Yun Xu. 2020. CCGraph: A PDG-based<br>Code Clone Detector with Approximate Graph Matching. In_Proceedings of the_|1121|
|1064||Query Reformulation for Code Search using Crowdsourced Knowledge._Empirical_||_35th IEEE/ACM International Conference on Automated Software Engineering (ASE)_.|1122|
|1065|[41]|_Software Engineering_24, 4 (2019), 1869–1924.<br> Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-||IEEE, 931–942.|1123|
|1066||work: BM25 and Beyond. _Foundations and Trends in Information Retrieval_3, 4|||1124|
|1067||(2009), 333–389. doi:10.1561/1500000019|||1125|
|1068|[42]|Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xi-<br>aoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom|||1126|
|1069||Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer,|||1127|
|1070||Aaron Grattafori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar,|||1128|
|1071||Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel<br>Synnaeve. 2023. Code Llama: Open Foundation Models for Code. _arXiv preprint_|||1129|
|1072||_arXiv:2308.12950_(2023).|||1130|
|1073|[43]|David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja<br>Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton,|||1131|
|1074||Yutian Chen, Timothy P. Lillicrap, Fan Hui, Laurent Sifre, George van den Driess-|||1132|
|1075||che, Thore Graepel, and Demis Hassabis. 2017. Mastering the game of Go without|||1133|
|1076|[44]|human knowledge. _Nature_550, 7676 (2017), 354–359. doi:10.1038/nature24270<br> Aryan Singhal, Rajat Ghosh, Ria Mundra, Harshil Dadlani, and Debojyoti Dutta.|||1134|
|1077||2025. Code2JSON: Can a Zero-Shot LLM Extract Code Features for Code RAG?.|||1135|
|1078||In_ICLR 2025 Workshop on Deep Learning for Code (DL4C)_. https://openreview.|||1136|
|1079|[45]|net/pdf/50ee288b55025f971ddbf1cb05b90a11dbb7feb7.pdf<br> Xiaozhuang Song, Shufei Zhang, and Tianshu Yu. 2025. ReKG-MCTS: Reinforcing|||1137|
|1080||LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree|||1138|
|1081|[46]|Search. In_Findings of the Association for Computational Linguistics: ACL 2025_.<br> Zeyu Sun, Qihao Zhu, Yingfei Xiong, Yican Sun, Lili Mou, and Lu Zhang. 2020.|||1139|
|1082||TreeGen: A Tree-Based Transformer Architecture for Code Generation. In_Pro-_|||1140|
|1083||_ceedings of the AAAI Conference on Artifcial Intelligence_, Vol. 34. 8984–8991.|||1141|
|1084|[47]|Hongyuan Tao, Ying Zhang, Zhenhao Tang, Hongen Peng, Xukun Zhu,<br>Bingchang Liu, Yingguang Yang, Ziyin Zhang, Zhaogui Xu, Haipeng Zhang,|||1142|
|1085||Linchao Zhu, Rui Wang, Hang Yu, Jianguo Li, and Peng Di. 2025. Code Graph|||1143|
|1086||Model (CGM): A Graph-Integrated Large Language Model for Repository-Level|||1144|
|1087|[48]|Software Engineering Tasks. _arXiv preprint arXiv:2505.16901_(2025).<br> Chaozheng Wang, Zhenghao Nong, Cuiyun Gao, Zongjie Li, Jichuan Zeng, Zhen-|||1145|
|1088||chang Xing, and Yang Liu. 2022. Enriching Query Semantics for Code Search|||1146|
|1089|[49]|with Reinforcement Learning. _Neural Networks_145 (2022), 22–32.<br> Xin Wang, Yasheng Wang, Yao Wan, Jiawei Wang, Pingyi Zhou, Li Li, Hao Wu,|||1147|
|1090||and Jin Liu. 2022. CODE-MVP: Learning to Represent Source Code from Multiple|||1148|
|1091||Views with Contrastive Pre-Training. In _Findings of the Association for Com-_|||1149|
|1092||_putational Linguistics: NAACL 2022_. Association for Computational Linguistics,<br>1066–1077.|||1150|
|1093|[50]|Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and|||1151|
|1094||Zibin Zheng. 2025. RLCoder: Reinforcement Learning for Repository-Level Code|||1152|
|1095||Completion. In_Proceedings of the 47th IEEE/ACM International Conference on_<br>_Software Engineering (ICSE)_. https://arxiv.org/abs/2407.19487|||1153|
|1096|[51]|Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Gra-|||1154|
|1097||ham Neubig, and Daniel Fried. 2025. CodeRAG-Bench: Can Retrieval Augment<br>Code Generation?. In_Findings of the Association for Computational Linguistics:_|||1155|
|1098||_NAACL 2025_. Association for Computational Linguistics, Albuquerque, New|||1156|
|1099||Mexico, 3199–3214. doi:10.18653/v1/2025.fndings-naacl.176|||1157|
|1100|[52]|Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettle-<br>moyer. 2020. Scalable Zero-shot Entity Linking with Dense Entity Retrieval. In|||1158|
|1101|||||1159|
|1102|||10||1160|



