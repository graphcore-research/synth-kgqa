# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import os
import json

# from datasets import load_dataset
import multiprocessing as mp

# import utils
from tqdm import tqdm

# from functools import partial
import numpy as np
import os.path as osp
import pickle


def build_data(args):
    """
    Extract the paths between question and answer entities from the dataset.
    """

    node_labels = np.load(
        osp.join(args.kg_path, "node_labels.npy"),
        allow_pickle=True,
    )
    relation_labels = np.load(
        osp.join(args.kg_path, "relation_labels.npy"),
        allow_pickle=True,
    )

    # input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d)

    print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # Load dataset
    # dataset = load_dataset(input_file, split=args.split)
    dataset = pickle.load(open(os.path.join(args.data_path, args.split), "rb"))
    results = []
    # for _ in range(args.n_epochs):
    # for sample in tqdm([dataset[i] for i in (np.random.permutation(range(len(dataset))) if args.shuffle else range(len(dataset)))]):
    #     results.extend(process_data(sample, node_labels, relation_labels, remove_duplicate=args.remove_duplicate, use_sp=args.use_sp))

    for sample in tqdm(dataset):
        results.extend(
            process_data(
                sample,
                node_labels,
                relation_labels,
                remove_duplicate=args.remove_duplicate,
                use_sp=args.use_sp,
            )
        )

    json.dump(results, open(os.path.join(output_dir, args.save_name), "w"))
    # with open(os.path.join(output_dir, args.save_name), 'w') as fout:
    #     with mp.Pool(args.n) as pool:

    #             for res in tqdm(pool.imap_unordered(partial(process_data, remove_duplicate=args.remove_duplicate, use_sp=args.use_sp), dataset), total=len(dataset)):
    #                 for r in res:
    #                     fout.write(json.dumps(r) + '\n')


def process_data(
    data, node_labels, relation_labels, remove_duplicate=False, use_sp=False
):
    # graph  =  utils.build_graph(data['graph'])
    # paths = utils.get_truth_paths(data['q_entity'], data['a_entity'], graph)
    # paths_list = []
    # for p in data['seed_answer_paths_sp' if use_sp else 'seed_answer_paths']:
    #     p_lab = [[node_labels[trip[0]], ["", "inverse of: "][trip[1] // len(relation_labels)] + relation_labels[trip[1] % len(relation_labels)], node_labels[trip[2]]] for trip in p]
    #     paths_list.append(p_lab)
    # paths = paths_list
    # result = []
    # # Split each Q-P pair into a single data
    # rel_paths = []
    # for path in paths:
    #     rel_path = [p[1] for p in path] # extract relation path
    #     if remove_duplicate:
    #         if tuple(rel_path) in rel_paths:
    #             continue
    #     rel_paths.append(tuple(rel_path))
    # for rel_path in rel_paths:
    #     if "paraphrased_question" in data:
    #         if np.random.rand() < 0.5:
    #             question = data["paraphrased_question"]
    #         else:
    #             question = data["question"]
    #     else:
    #         question = data['question']

    #     result.append({"question": question, "path": rel_path})
    # return result

    result = []
    for i, path_list in enumerate(
        data["seed_answer_paths_sp" if use_sp else "seed_answer_paths"].values()
    ):
        # path = path_list[np.random.randint(len(path_list))]
        for path in path_list:
            p_lab = [
                [
                    node_labels[trip[0]],
                    ["", "inverse of: "][trip[1] // len(relation_labels)]
                    + relation_labels[trip[1] % len(relation_labels)],
                    node_labels[trip[2]],
                ]
                for trip in path
            ]
            rel_path = [t[1] for t in p_lab]
            if "paraphrased_question" in data and np.random.rand() < 0.5:
                question = data["paraphrased_question"]
            else:
                question = data["question"]
            result.append(
                {
                    "id": data.get("id", None),
                    "seed_answ_pair": i,
                    "question": question,
                    "path": rel_path,
                }
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument("--d", "-d", type=str, default="synthetic")
    parser.add_argument(
        "--kg-path", type=str, help="path to directory containing the KG"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="datasets/AlignData")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--n", "-n", type=int, default=1)
    parser.add_argument("--remove_duplicate", action="store_true")
    parser.add_argument(
        "--use_sp", action="store_true", help="use SP instead of GT paths"
    )
    args = parser.parse_args()

    if args.save_name == "":
        args.save_name = (
            args.d + "_" + args.split + ("_sp" if args.use_sp else "") + ".jsonl"
        )

    build_data(args)
