#!/bin/bash

bash scripts/build_graph_index.sh
bash scripts/train_kg_specialized_llm.sh
bash scripts/graph_constrained_decoding.sh
bash scripts/graph_inductive_reasoning.sh
bash scripts/train_kg_specialized_llm_sp.sh
bash scripts/graph_constrained_decoding_sp.sh
bash scripts/graph_inductive_reasoning_sp.sh