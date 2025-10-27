# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Linhao Luo (Raymond)


import os
import argparse
import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
import json
import pickle
import numpy as np
import os.path as osp
from functools import partial
from src.utils.graph_utils import build_graph, dfs, get_truth_paths

def process(sample, undirected, node_labels, relation_labels, use_sp=False):
    # gph = [[node_labels[h], relation_labels[r], node_labels[t]] for h,r,t in sample["answer_subgraph"]] # use GT subgraph
    # gph = [[node_labels[h], relation_labels[r], node_labels[t]] for h,r,t in zip(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"])] # use shortest path subgraph
    # graph = build_graph(gph, undirected=undirected)
    # start_nodes = [node_labels[x] for x in sample['seed_nodes_id']]
    # answer_nodes = [node_labels[sample['answer_node_id']]]
    # paths_list = get_truth_paths(start_nodes, answer_nodes, graph)
    # paths_list = []
    # for p in sample['seed_answer_paths_sp' if use_sp else 'seed_answer_paths']:
    #     p_lab = [[node_labels[trip[0]], ["", "inverse of: "][trip[1] // len(relation_labels)] + relation_labels[trip[1] % len(relation_labels)], node_labels[trip[2]]] for trip in p]
    #     paths_list.append(p_lab)
    # sample['ground_truth_paths'] = paths_list
    # return {k: v for k,v in sample.items() if k in ["id", "question", "paraphrased_question", "seed_nodes", "seed_nodes_id", "answer_node", "answer_node_id", "all_answers", "answer_subgraph", "full_subgraph", "ground_truth_paths"]}

    result = []
    for i, (seed_ans, path_list) in enumerate(sample['seed_answer_paths_sp' if use_sp else 'seed_answer_paths'].items()):
        # path = path_list[np.random.randint(len(path_list))]
        for path in path_list:
            p_lab = [[node_labels[trip[0]], ["", "inverse of: "][trip[1] // len(relation_labels)] + relation_labels[trip[1] % len(relation_labels)], node_labels[trip[2]]] for trip in path]
            result.append({"id": sample.get("id", None), "seed_answ_pair": i, "seed_nodes_id": [seed_ans[0]], "question": sample["question"], "paraphrased_question": sample["paraphrased_question"], "ground_truth_paths": [p_lab]})
    return result
    

def index_graph(args):

    node_labels = np.load(
        osp.join(args.kg_path, "node_labels.npy"),
        allow_pickle=True,
    )
    relation_labels = np.load(
        osp.join(args.kg_path, "relation_labels.npy"),
        allow_pickle=True,
    )

    # input_file = os.path.join(args.data_path, args.d)
    # data_path = f"{args.d}_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.output_path, args.d, args.split)
    # Load dataset
    # dataset = load_dataset(input_file, split=args.split)

    # dataset = json.load(open(os.path.join(args.data_path, f"{args.split}.jsonl")))
    dataset = pickle.load(open(os.path.join(args.data_path, args.split), "rb"))
    
    # dataset = dataset.map(process, num_proc=args.n, fn_kwargs={'K': args.K})
    # dataset.select_columns(['id', 'paths']).save_to_disk(output_dir)
    results = []
    # with Pool(args.n) as p:
    #     for res in tqdm.tqdm(p.imap_unordered(partial(process, undirected = args.undirected, node_labels = node_labels, relation_labels = relation_labels, use_sp = args.use_sp), dataset), total=len(dataset)):
    #         results.append(res)
    for sample in tqdm.tqdm(dataset):
        results.extend(process(sample, args.undirected, node_labels, relation_labels, use_sp=args.use_sp))
    
    index_dataset = Dataset.from_list(results)
    index_dataset.save_to_disk(output_dir)
        
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path')
    argparser.add_argument('--kg-path', type=str)
    argparser.add_argument('--d', '-d', type=str, default='synthetic_kgqa')
    argparser.add_argument('--split', type=str, default='train')
    argparser.add_argument('--output_path', type=str, default='data/shortest_path_index')
    argparser.add_argument('--undirected', action='store_true', help='whether the graph is undirected')
    argparser.add_argument('--n', type=int, default=1, help='number of processes')
    argparser.add_argument('--use_sp', action='store_true', help='use SP instead of GT paths')
    
    args = argparser.parse_args()
    
    index_graph(args)