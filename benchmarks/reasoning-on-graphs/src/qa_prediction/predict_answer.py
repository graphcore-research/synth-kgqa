# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)


import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
import numpy as np
import os.path as osp
import re
import pickle


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"][
            :4
        ]  ## MODIIIIIIIIIIIIIIIIIIIIIIIIIIFICATO
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    # qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    qa_dataset = [find_rule(x) for x in qa_dataset]
    if filter_empty:
        # qa_dataset = qa_dataset.filter(
        #     lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        # )
        qa_dataset = [x for x in qa_dataset if len(x["ground_paths"]) > 0]
    return qa_dataset


def prediction(
    data, processed_list, input_builder, model, node_labels, relation_labels
):
    question = data["question"]
    answer = node_labels[data["answer_node_id"]]
    data["graph"] = [
        [node_labels[h], relation_labels[r], node_labels[t]]
        for h, r, t in zip(data["h_id_list"], data["r_id_list"], data["t_id_list"])
    ]
    data["q_entity"] = [node_labels[x] for x in data["seed_nodes_id"]]
    data["choices"] = []
    id = data["id"]
    if id in processed_list:
        return None
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    input, lists_of_paths = input_builder.process_input(data)
    prediction = model.generate_sentence(input)
    if prediction is None:
        return None
    result = {
        k: data[k]
        for k in [
            "id",
            "question",
            "seed_nodes",
            "seed_nodes_id",
            "answer_node",
            "answer_node_id",
            "sparql_query",
            "all_answers",
            "answer_subgraph",
            "full_subgraph",
            "n_hops",
            "graph_template",
            "answer_triples",
            "redundant",
        ]
    }
    result.update(
        {
            "graphrag_paths": lists_of_paths,
            "graphrag_answer": prediction,
            "predicted_paths": data["predicted_paths"],
            "ground_truth_paths": data["ground_paths"],
            "ground_truth": answer,
            "input": input,
        }
    )
    return result


def main(args, LLM):
    kg_path = "../../data/ogbl-wikikg2"

    node_labels = np.load(
        osp.join(kg_path, "node_labels.npy"),
        allow_pickle=True,
    )
    relation_labels = np.load(
        osp.join(kg_path, "relation_labels.npy"),
        allow_pickle=True,
    )

    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    # dataset = load_dataset(input_file, split=args.split)
    dataset = pickle.load(open(input_file, "rb"))

    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"

    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            # tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                        node_labels=node_labels,
                        relation_labels=relation_labels,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(
                data, processed_list, input_builder, model, node_labels, relation_labels
            )
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        type=str,
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
