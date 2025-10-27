# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from synth_kgqa import graph_sampling, llm, parse, prompts


def main(args):
    if args.exclude_nodes:
        exclude_nodes = np.load(args.exclude_nodes)
    else:
        exclude_nodes = np.array([], dtype=np.int32)
    graph = graph_sampling.Graph.from_directory(
        args.kg_path,
        exclude_rels=args.exclude_rels,
        exclude_nodes=exclude_nodes,
    )
    dataset = graph_sampling.GraphSamplingDataset(
        graph=graph,
        subgraph_sampling_method=args.subgraph_sampling_method,
        num_nodes=args.num_subgraph_nodes,
        num_edges=args.num_subgraph_edges,
        label_sep_token=args.label_sep_token,
        triple_sep_token=args.triple_sep_token,
        random_seed=args.random_seed,
        degree_bias=args.degree_bias,
        seed_graphs=[],
    )

    llm_api = llm.LLMAPI(args.api_name, model=args.llm)
    rng = np.random.default_rng(seed=args.random_seed)

    os.makedirs(args.save_path, exist_ok=True)
    graph_samples = []
    llm_outputs = []
    for n in tqdm(range(args.num_samples)):
        seed_node_idx = rng.integers(graph.num_connected_nodes)
        sample = dataset[seed_node_idx]
        graph_samples.append(sample)
        llm_out = llm_api(
            prompt=prompts.n_hop_prompt(args.num_edges, sample["flattened_graph"])
        )
        llm_outputs.append(llm_out)

        if (n % 100 == 0) or (n == args.num_samples - 1):
            with open(osp.join(args.save_path, "samples.pkl"), "wb") as f:
                pickle.dump(graph_samples, f)

            with open(osp.join(args.save_path, "llm_outputs.pkl"), "wb") as f:
                pickle.dump(llm_outputs, f)


if __name__ == "__main__":
    args = parse.parse_generation_args()
    main(args)
