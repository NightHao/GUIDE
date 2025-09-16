# Research_CODE Project

A comprehensive entity graph construction and query processing system for technical document analysis.

## Project Structure

```
GUIDE/
├── src/                    # Source code
│   ├── build_alias_list.py           # Alias dictionary builder
│   ├── entities_to_graph.py          # Entity graph construction
│   ├── process_entity_graph.py       # Graph processing & abbreviation handling
│   ├── eg_node_description_optimizer.py  # Description optimization
│   ├── agentic_flow_construction.py  # Multi-agent workflow
│   ├── query_processor.py           # Main query processing engine
│   ├── prompt_eng.py               # Advanced prompt engineering
│   ├── merge_prompts.py            # Prompt merging utilities
│   ├── main.py                     # Simple prompt runner
│   ├── run_final_prompt.py         # Batch prompt processor
│   ├── test_single_file.py         # Single file testing
│   └── agentic_flow_eval.py        # Evaluation framework
├── data/
│   ├── input/                      # Input data files
│   │   ├── entities_result.json           # Raw entity extraction results
│   │   ├── retrieved_chunks_15.json       # Question-chunk mappings
│   │   ├── final_prompt.json             # Final prompt templates
│   │   └── complete_prompts/             # Source prompt files
│   ├── intermediate/               # Processing intermediate files
│   │   ├── entity_graph.json            # Initial entity graph
│   │   ├── processed_entity_graph.json  # Processed graph
│   │   ├── alias_dict.json              # Alias mappings
│   │   ├── log.json                     # Processing logs
│   │   ├── entities_chunks.json         # Entity chunks
│   │   └── merged_prompts/              # Merged prompt files
│   └── output/                     # Final outputs
│       ├── optimized_entity_graph.json  # Optimized graph
│       ├── o1_answers.json              # Generated answers
│       ├── new_prompts/                 # Processed prompts
│       └── evaluation_results/          # Evaluation outputs
├── docs/                          # Documentation
│   ├── Research_CODE_Documentation.md   # Detailed file documentation
│   └── system_flowchart.md             # System flow diagram
├── tests/                         # Test files and outputs
├── config/                        # Configuration files
│   └── requirements.txt                 # Python dependencies
└── README.md                      # This file
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r config/requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Prepare Input Data**:
   - Place `entities_result.json` in `data/input/`
   - Place `retrieved_chunks_15.json` in `data/input/`

4. **Run the Pipeline**:
   ```bash
   # Step 1: Build entity graph
   python src/entities_to_graph.py

   # Step 2: Process graph (abbreviations, patterns)
   python src/process_entity_graph.py

   # Step 3: Optimize descriptions
   python src/eg_node_description_optimizer.py

   # Step 4: Build aliases
   python src/build_alias_list.py

   # Step 5: Process queries
   python src/query_processor.py
   ```

## System Overview

This system processes technical documents through a multi-stage pipeline:

1. **Entity Graph Construction**: Converts raw entity extraction results into a structured graph
2. **Graph Processing**: Identifies abbreviations, merges related entities, and optimizes structure
3. **Query Processing**: Handles user questions using agentic workflows and entity context
4. **Evaluation**: Comprehensive evaluation framework for system performance

See `docs/system_flowchart.md` for detailed flow diagram and `docs/Research_CODE_Documentation.md` for comprehensive documentation.

## Key Features

- **Abbreviation Handling**: Automatic identification and merging of abbreviations with full names
- **Agentic Workflows**: Multi-agent system for intelligent query processing
- **Graph Optimization**: Sentence clustering and merging for improved performance
- **Comprehensive Evaluation**: Multiple evaluation metrics and comparison tools
- **Flexible Configuration**: Support for different graph types and processing parameters

## Dependencies

- Python 3.8+
- OpenAI API access
- See `config/requirements.txt` for full dependency list