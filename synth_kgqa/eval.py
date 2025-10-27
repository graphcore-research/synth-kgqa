# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import argparse
import json
import os.path as osp
import pickle
import re

import numpy as np
from tqdm import tqdm

from synth_kgqa import llm, prompts


def new_exact_match(response, answers, retriever=False):
    clean_result = response.strip().replace(" ", "").lower()
    find_all = re.findall(r"\{(.*?)\}", clean_result)
    if len(find_all) > 0:
        if find_all[-1] not in ["yes", "no"]:
            final_ans = find_all[-1]
        else:
            final_ans = clean_result
    else:
        final_ans = clean_result
    for answer in answers:
        clean_answer = answer.strip().replace(" ", "").lower()
        if (
            final_ans == clean_answer
            or clean_answer in final_ans
            or final_ans in clean_answer
        ):
            return True
    return False


def is_complete(query: str, subgraph, llm_api, node_labels, relation_labels):
    knowledge = ""
    for h, r, t in subgraph:
        knowledge += (
            f"{node_labels[h]}, " f"{relation_labels[r]}, " f"{node_labels[t]}\n"
        )
    out = llm_api(prompts.evaluate_complete_prompt(query, knowledge))
    if "yes" in out.lower():
        return True, out
    return False, out


def main(args):
    llm_api = llm.LLMAPI(args.api_name, model=args.llm)

    node_labels = np.load(osp.join(args.kg_path, "node_labels.npy"), allow_pickle=True)
    relation_labels = np.load(
        osp.join(args.kg_path, "relation_labels.npy"), allow_pickle=True
    )
    node_qids = np.load(
        osp.join(args.kg_path, "node_ids.npy"),
        allow_pickle=True,
    )

    rev_lkp_ent = {ent: idx for idx, ent in enumerate([int(x[1:]) for x in node_qids])}

    if ".pkl" in args.dataset_path:
        ds = pickle.load(open(args.dataset_path, "rb"))
    else:
        ds = json.load(open(args.dataset_path, "rb"))
    ds_out = []

    pbar = tqdm(
        enumerate(ds),
        total=len(ds),
    )

    good = []
    for i, dp in pbar:
        if i > 0:
            pbar.set_description(f"Acceptance rate: {(100 * len(good)/i):.1f}%")
        dp_out = {"id": i}
        dp_out.update({k: v for k, v in dp.items() if "seed_answer_paths" not in k})
        for query, suffix in [
            (dp["question"], ""),
            # (dp["paraphrased_question"], "_paraphrased"),
        ]:
            answ_nodes = [
                rev_lkp_ent.get(int(qid[1:]), -1)
                for qid in dp["all_answers"]
                if qid[0] == "Q"
            ]
            answ_nodes = [node_labels[x] for x in answ_nodes if x > -1]
            if not args.llm_only:
                if args.use_sp:
                    graph = dp["full_subgraph_sp"]
                else:
                    graph = dp["answer_subgraph"]
                for k in range(args.n_tries):
                    _, subgraph_answer = is_complete(
                        query, graph, llm_api, node_labels, relation_labels
                    )
                    dp_out["graphrag_answer" + suffix + f"_{k}"] = subgraph_answer

                    graphrag_em = new_exact_match(subgraph_answer, answ_nodes)
                    dp_out["graphrag_em" + suffix + f"_{k}"] = graphrag_em

            for k in range(args.n_tries):
                io_out = llm_api(prompts.io_prompt(query))
                dp_out["io_answer" + suffix + f"_{k}"] = io_out
                cot_out = llm_api(prompts.cot_prompt(query))
                dp_out["cot_answer" + suffix + f"_{k}"] = cot_out
                dp_out["io_em" + suffix + suffix + f"_{k}"] = new_exact_match(
                    io_out, answ_nodes
                )
                dp_out["cot_em" + suffix + suffix + f"_{k}"] = new_exact_match(
                    cot_out, answ_nodes
                )

        ds_out.append(dp_out)

        good_lab = "cot_em" if args.llm_only else "graphrag_em"
        if np.all([dp_out[good_lab + f"_{k}"] for k in range(args.n_tries)]):
            good.append(i)

        if i % 50 == 0 or i == len(ds) - 1:
            with open(args.output_file, "w") as f:
                json.dump(ds_out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg-path",
        type=str,
    )
    parser.add_argument("--api-name", type=str, default="openai")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("-d", "--dataset_path", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("--use_sp", action="store_true")
    parser.add_argument("--llm_only", action="store_true")
    parser.add_argument("--n_tries", type=int, default=2)
    args = parser.parse_args()
    main(args)
