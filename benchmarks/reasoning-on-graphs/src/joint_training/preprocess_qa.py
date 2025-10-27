# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets
from qa_prediction.build_qa_input import PromptBuilder
import pickle
from tqdm import tqdm
import numpy as np
import os.path as osp

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)

USE_SP = False


save_dir = "datasets/joint_training/qa"
prompt_path = "prompts/llama2_predict.txt"
split = "train_def_scores.pkl"
model_max_length = 2048 - 200
data_list = ["synthetic_def"]
data_path = "../../GTSQA"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
prompter = InstructFormater(prompt_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
)


# Load prompt template
input_builder = PromptBuilder(
    prompt_path,
    add_rule=True,
    use_true=True,
    maximun_token=model_max_length,
    tokenize=lambda x: len(tokenizer.tokenize(x)),
)


def formatting_prompts_func(
    example, node_labels, relation_labels, rev_lkp_ent, use_sp=False
):
    # answ_nodes = [
    #         rev_lkp_ent.get(int(qid[1:]), -1)
    #         for qid in example["all_answers"]
    #         if qid[0] == "Q"
    #     ]
    # output_label = "\n".join([node_labels[x] for x in [a for a in answ_nodes if a > -1]])
    output_label = node_labels[example["answer_node_id"]]
    # Find ground-truth paths for each Q-P pair
    # graph = build_graph(example["graph"])
    # paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    # ground_paths = set()
    # for path in paths:
    #     ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    # example["ground_paths"] = list(ground_paths)
    paths_list = []
    for p in example["seed_answer_paths_sp" if use_sp else "seed_answer_paths"]:
        p_lab = [
            [
                node_labels[trip[0]],
                ["", "inverse of: "][trip[1] // len(relation_labels)]
                + relation_labels[trip[1] % len(relation_labels)],
                node_labels[trip[2]],
            ]
            for trip in p
        ]
        paths_list.append(p_lab)
    ground_paths = set()
    for path in paths_list:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    example["ground_paths"] = list(ground_paths)

    gt_graph = [
        [node_labels[h], relation_labels[r], node_labels[t]]
        for h, r, t in example["answer_subgraph"]
    ]
    # graph = [[node_labels[h], relation_labels[r], node_labels[t]] for h,r,t in zip(data["h_id_list"], data["r_id_list"], data["t_id_list"])]

    output_text = (
        input_builder.process_input(
            {
                "question": example["question"],
                "q_entity": [node_labels[x] for x in example["seed_nodes_id"]],
                "a_entity": node_labels[example["answer_node_id"]],
                "graph": gt_graph,
                "ground_paths": example["ground_paths"],
                "choices": [],
            }
        )
        + " "
        + output_label
        + tokenizer.eos_token
    )
    return {"text": output_text}


kg_path = "../../data/ogbl-wikikg2"

node_labels = np.load(
    osp.join(kg_path, "node_labels.npy"),
    allow_pickle=True,
)
relation_labels = np.load(
    osp.join(kg_path, "relation_labels.npy"),
    allow_pickle=True,
)

node_qids = np.load(
    osp.join(kg_path, "node_ids.npy"),
    allow_pickle=True,
)
rev_lkp_ent = {ent: idx for idx, ent in enumerate([int(x[1:]) for x in node_qids])}

for data_name in data_list:
    if os.path.exists(os.path.join(save_dir, data_name)) == False:
        os.makedirs(os.path.join(save_dir, data_name))
    input_file = os.path.join(data_path, data_name)
    # train_dataset = datasets.load_dataset(input_file, split="train")
    train_dataset = pickle.load(open(os.path.join(data_path, split), "rb"))
    save_path = os.path.join(save_dir, data_name, data_name + "_train.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")

    # train_dataset = train_dataset.map(
    #     formatting_prompts_func,
    #     remove_columns=train_dataset.column_names,
    #     num_proc=N_CPUS,
    # )
    # train_dataset.to_json(save_path, orient="records", lines=True)

    results = []
    for sample in tqdm(train_dataset):
        results.append(
            formatting_prompts_func(
                sample, node_labels, relation_labels, rev_lkp_ent, use_sp=USE_SP
            )
        )

    json.dump(results, open(save_path, "w"))
