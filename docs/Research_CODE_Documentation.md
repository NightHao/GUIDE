# Research_CODE Python Files Documentation

This document explains how each Python file in the Research_CODE directory works, including their input/output data flow and functionality.

## 1. build_alias_list.py

**Purpose**: Builds a bidirectional alias dictionary from abbreviation data.

**Functionality**:
- Loads abbreviation dictionary from `log.json`
- Creates bidirectional mapping between abbreviations and full names
- Normalizes full names with proper capitalization

**Input**:
- `./log.json` (contains abbreviation dictionary data)

**Output**:
- `./alias_dict.json` (bidirectional alias dictionary)

**Key Features**:
- Handles hyphenated words in normalization
- Creates both abbreviation→full_name and full_name→abbreviation mappings

---

## 2. entities_to_graph.py

**Purpose**: Processes entity extraction results and builds a NetworkX graph from entities and relationships.

**Functionality**:
- Parses entity and relationship records from LLM responses
- Builds undirected graph with entity nodes and relationship edges
- Handles merging of duplicate entities and relationships
- Converts NetworkX graph to dictionary format

**Input**:
- `./entities_result.json` (array of objects with "response" key containing entity extraction results)

**Output**:
- `./entity_graph.json` (entity graph in dictionary format)

**Key Features**:
- Supports both entity and relationship record types
- Merges descriptions and source IDs for duplicate entities
- Processes tuple-delimited records with configurable delimiters

---

## 3. process_entity_graph.py

**Purpose**: Advanced entity graph processing including abbreviation extraction, pattern recognition, and node merging.

**Functionality**:
- Identifies abbreviations using GPT-4o-mini
- Extracts abbreviation-to-full-name patterns using o3-mini
- Clusters similar patterns using embeddings
- Merges abbreviation nodes with their full name counterparts
- Builds comprehensive abbreviation dictionary

**Input**:
- `./entity_graph.json` (entity graph from entities_to_graph.py)
- Environment variables (OpenAI API keys)

**Output**:
- `./processed_entity_graph.json` (processed graph with merged abbreviations)
- `./log.json` (processing logs and intermediate results)

**Key Features**:
- Asynchronous processing for efficiency
- Pattern clustering using cosine similarity
- Regex pattern generation for abbreviation extraction
- Comprehensive logging of processing steps

---

## 4. agentic_flow_construction.py

**Purpose**: Constructs an agentic workflow for processing questions about entities in the graph.

**Functionality**:
- Implements BFS traversal for entity graph exploration
- Builds entity lists with relationship information
- Creates multi-agent workflow using LangGraph
- Handles entity name validation and fuzzy matching
- Processes background chunks for context

**Input**:
- Entity graph files (configurable path)
- `./alias_dict.json` (for entity name mapping)
- User questions with background chunks

**Output**:
- Processed entity information in JSON format
- Entity chunks with contextual information

**Key Features**:
- Distance-based subgraph extraction
- Interactive user confirmation for entity matching
- Asynchronous chunk processing
- Support for common vs unique entity analysis

---

## 5. merge_prompts.py

**Purpose**: Merges complete prompts with final prompt content for question answering.

**Functionality**:
- Extracts questions from prompt filenames
- Maps simplified questions to formal versions
- Merges content from multiple prompt sources
- Reformats prompt structure for consistency

**Input**:
- `/home/yeskey525/Research_CODE/experiments/golden_answers_exe/complete_prompts/*.txt`
- `./final_prompt.json`

**Output**:
- `./merged_prompts/*.txt` (merged prompt files)

**Key Features**:
- Automatic question extraction from filenames
- Content restructuring and reformatting
- Support for special protocol command patterns

---

## 6. main.py

**Purpose**: Simple script to run final prompt through LLM and format output.

**Functionality**:
- Loads prompt from file
- Processes through LLM
- Formats response with markdown headers

**Input**:
- `final_prompt.txt`

**Output**:
- Formatted LLM response to stdout

**Key Features**:
- Basic prompt-to-answer pipeline
- Automatic markdown formatting

---

## 7. run_final_prompt.py

**Purpose**: Batch processes multiple questions from final_prompt.json using LLM with checkpointing.

**Functionality**:
- Loads questions and prompts from JSON
- Processes each question through o1 model
- Implements checkpointing to resume interrupted processing
- Formats answers with markdown

**Input**:
- `final_prompt.json` (questions and prompts)
- `o1_answers.json` (existing answers for checkpointing)

**Output**:
- `o1_answers.json` (LLM answers for all questions)

**Key Features**:
- Incremental processing with save-after-each-answer
- Skip already processed questions
- Error handling and recovery

---

## 8. test_single_file.py

**Purpose**: Tests prompt engineering on a single file.

**Functionality**:
- Processes a specific prompt file
- Saves processed output for examination

**Input**:
- `/home/yeskey525/Research_CODE/merged_prompts/What is EIM?.txt`

**Output**:
- `/home/yeskey525/Research_CODE/sample_processed_eim.txt`

**Key Features**:
- Single-file testing for prompt engineering validation

---

## 9. prompt_eng.py

**Purpose**: Advanced prompt engineering with structured template generation.

**Functionality**:
- Parses complex prompt files with multiple sections
- Extracts entities, tables, figures, and contextual information
- Creates structured prompts using professional templates
- Handles deduplication of repeated content

**Input**:
- Prompt files from various directories (configurable)

**Output**:
- Structured prompt files with technical documentation format

**Key Features**:
- Section-based content extraction
- Duplicate content removal
- Professional technical documentation templates
- Support for entity chunks, figures, and tables

---

## 10. eg_node_description_optimizer.py

**Purpose**: Optimizes entity graph node descriptions by clustering and merging similar sentences.

**Functionality**:
- Splits descriptions into sentences
- Uses embeddings to cluster similar sentences
- Employs LLM to merge similar sentence clusters
- Processes entire graphs or individual nodes
- Compares Q&A performance on original vs optimized descriptions

**Input**:
- `./processed_entity_graph.json` (or other entity graph files)

**Output**:
- `./optimized_entity_graph.json` (graph with optimized descriptions)

**Key Features**:
- Asynchronous batch processing
- Configurable similarity thresholds
- Iterative optimization with limits
- Q&A comparison functionality

---

## 11. query_processor.py

**Purpose**: Main query processing engine that handles question answering using the entity graph.

**Functionality**:
- Processes user questions through agentic flow
- Extracts user intentions (general, comparison, commonality)
- Retrieves relevant chunks for questions
- Generates comprehensive answers using entity context
- Manages entity graph extensions and filtering

**Input**:
- `./retrieved_chunks_15.json` (question-to-chunk mappings)
- `./entities_chunks.json` (processed entity chunks)
- Entity graph files (configurable)

**Output**:
- Detailed answers to user questions
- Updated entities_chunks.json with new question data

**Key Features**:
- Intent classification for different question types
- Fuzzy matching for question retrieval
- Entity graph extension for enhanced context
- Common vs unique entity analysis

---

## 12. agentic_flow_eval.py

**Purpose**: Comprehensive evaluation framework for the agentic flow system.

**Functionality**:
- Generates answers using various methods (with/without agentic flow)
- Evaluates answer correctness using DeepEval metrics
- Performs LLM-based judgments between different approaches
- Supports multiple evaluation scenarios and datasets

**Input**:
- Various answer files (e.g., `./experiments/Answers/*.json`)
- Ground truth files (e.g., `./experiments/ground_truth_QA.json`)
- Prompt files from different directories

**Output**:
- Evaluation results with correctness scores
- LLM judgment comparisons
- Performance metrics and statistics

**Key Features**:
- Multiple evaluation methodologies
- Checkpointing for long evaluations
- Statistical analysis of results
- Support for different LLM models and approaches

---

## Data Flow Summary

The files work together in the following pipeline:

1. **Entity Extraction → Graph Building**: `entities_to_graph.py` converts raw entity extraction results into a graph
2. **Graph Processing**: `process_entity_graph.py` enhances the graph by merging abbreviations and extracting patterns
3. **Graph Optimization**: `eg_node_description_optimizer.py` optimizes node descriptions for better performance
4. **Alias Building**: `build_alias_list.py` creates bidirectional alias mappings
5. **Prompt Engineering**: `prompt_eng.py` and `merge_prompts.py` prepare structured prompts
6. **Query Processing**: `query_processor.py` and `agentic_flow_construction.py` handle question answering
7. **Evaluation**: `agentic_flow_eval.py` evaluates system performance

### Key Directories Used:
- `./` (current directory) - Main data files and outputs
- `./experiments/` - Experimental data and results
- `./merged_prompts/` - Merged prompt files
- `./new_prompts/` - Processed prompt files
- `./experiments/golden_answers_exe/complete_prompts/` - Source prompt files