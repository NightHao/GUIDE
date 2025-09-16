# Input Data Directory

This directory contains the input data files required for the system to run.

## ✅ Available Files (Already Copied)

### Core Input Files

1. **entities_result.json** ✅
   - Raw entity extraction results from LLM processing
   - Format: Array of objects with "response" key containing entity/relationship data
   - Used by: `entities_to_graph.py`
   - Size: ~4.2MB

2. **retrieved_chunks_15.json** ✅
   - Question-to-background-chunk mappings
   - Format: Array of objects with "question" and "prompt" keys
   - Used by: `query_processor.py`
   - Size: ~382KB

3. **final_prompt.json** ✅
   - Final prompt templates for question processing
   - Format: Dictionary mapping questions to prompt templates
   - Used by: `merge_prompts.py`, `run_final_prompt.py`
   - Size: ~1.8MB

### Prompt Files

4. **merged_prompts/** ✅
   - Directory containing merged prompt files
   - Format: Text files with structured prompt content
   - Used by: `prompt_eng.py`
   - Contains: 30+ prompt files

5. **complete_prompts/** ✅
   - Directory containing complete individual prompt files
   - Format: Text files with comprehensive prompt content
   - Used by: `merge_prompts.py`, `prompt_eng.py`
   - Contains: Source prompt files

## Current File Structure

```bash
data/input/
├── entities_result.json          # ✅ 4.2MB - Entity extraction results
├── retrieved_chunks_15.json      # ✅ 382KB - Question-chunk mappings
├── final_prompt.json            # ✅ 1.8MB - Final prompt templates
├── merged_prompts/              # ✅ Directory with merged prompts
│   ├── What is EIM?.txt
│   ├── What is SLAC?.txt
│   └── ... (30+ prompt files)
└── complete_prompts/            # ✅ Directory with complete prompts
    ├── What is EIM?.txt
    ├── What is SLAC?.txt
    └── ... (source prompt files)
```

## Environment Variables

Make sure to set your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Data Sources

These files are typically generated from:
- Entity extraction pipelines (for entities_result.json)
- Document retrieval systems (for retrieved_chunks_15.json)
- Manual prompt curation (for prompt files)