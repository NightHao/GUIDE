# Research_CODE System Flow Chart

```mermaid
graph TD
    %% Input Data
    A[entities_result.json] --> B[entities_to_graph.py]
    ENV[Environment Variables<br/>OpenAI API Keys] --> C[process_entity_graph.py]

    %% Core Processing Pipeline
    B --> |entity_graph.json| C[process_entity_graph.py]
    C --> |processed_entity_graph.json| D[eg_node_description_optimizer.py]
    C --> |log.json| E[build_alias_list.py]
    E --> |alias_dict.json| F[agentic_flow_construction.py]
    D --> |optimized_entity_graph.json| F

    %% Prompt Engineering Pipeline
    G[complete_prompts/*.txt] --> H[merge_prompts.py]
    I[final_prompt.json] --> H
    H --> |merged_prompts/*.txt| J[prompt_eng.py]
    J --> |new_prompts/*.txt| K[run_final_prompt.py]

    %% Query Processing Pipeline
    L[retrieved_chunks_15.json] --> M[query_processor.py]
    F --> |Agentic Flow| M
    M --> |entities_chunks.json| N[Answer Generation]

    %% Evaluation Pipeline
    O[Ground Truth Data] --> P[agentic_flow_eval.py]
    N --> P
    K --> |Generated Answers| P
    P --> Q[Evaluation Results]

    %% Simple Processing
    R[final_prompt.txt] --> S[main.py]
    S --> T[Formatted Output]

    %% Testing
    U[Sample Files] --> V[test_single_file.py]
    V --> W[Test Output]

    %% Styling
    classDef inputFile fill:#e1f5fe
    classDef processFile fill:#f3e5f5
    classDef outputFile fill:#e8f5e8
    classDef testFile fill:#fff3e0

    class A,ENV,G,I,L,O,R,U inputFile
    class B,C,D,E,F,H,J,K,M,P,S,V processFile
    class N,Q,T,W outputFile
```

## Data Flow Description

### 1. Entity Graph Construction & Processing
```
entities_result.json → entities_to_graph.py → entity_graph.json
                                           ↓
                    process_entity_graph.py → processed_entity_graph.json
                                           ↓
                    eg_node_description_optimizer.py → optimized_entity_graph.json
```

### 2. Alias & Mapping
```
log.json → build_alias_list.py → alias_dict.json
```

### 3. Prompt Engineering
```
complete_prompts/*.txt + final_prompt.json → merge_prompts.py → merged_prompts/*.txt
                                                              ↓
                                           prompt_eng.py → new_prompts/*.txt
```

### 4. Query Processing
```
retrieved_chunks_15.json + optimized_entity_graph.json + alias_dict.json
                                           ↓
                    query_processor.py + agentic_flow_construction.py
                                           ↓
                                   entities_chunks.json
```

### 5. Answer Generation & Evaluation
```
new_prompts/*.txt → run_final_prompt.py → o1_answers.json
                                       ↓
Ground Truth + Generated Answers → agentic_flow_eval.py → Evaluation Results
```