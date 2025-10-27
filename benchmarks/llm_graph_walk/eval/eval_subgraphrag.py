# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import argparse
import json
import pickle
import time
import random

import numpy as np
import torch
from tqdm import tqdm
from utils import exact_match

from llm_graph_walk import graph, llm, prompts, sample_subgraphrag, text_encoder


class RetrieverDatasetEval(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            preprocessed_data = pickle.load(f)
        self.preprocessed_data = preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        sample = self.preprocessed_data[idx]
        return (
            sample["question"],
            torch.tensor(sample["h_id_list"]),
            torch.tensor(sample["r_id_list"]),
            torch.tensor(sample["t_id_list"]),
            len(sample["non_text_entity_list"]),
            torch.tensor(sample["topic_entity_id"]),
        )


def main(args):
    chkpt = torch.load(args.retriever_path)
    config = chkpt["config"]

    t_enc = text_encoder.TextEncoderSubgraphRAG(config.text_encoder_path, device="cuda")

    if args.use_full_wikidata:
        pass
        # kg = graph.KGInterfaceFromWikidata(args.wikidata_server_urls)
    else:
        node_labels = np.load(
            args.wikikg_dir + "/node_labels.npy",
            allow_pickle=True,
        )
        relation_labels = np.load(
            args.wikikg_dir + "/relation_labels.npy",
            allow_pickle=True,
        )
        edge_ids = np.load(
            args.wikikg_dir + "/edge_ids.npy",
            allow_pickle=True,
        )
        relation_types = np.load(
            args.wikikg_dir + "/relation_types.npy",
            allow_pickle=True,
        )

        knowledge_graph = graph.Graph(
            edge_ids,
            relation_types,
            node_labels,
            relation_labels,
            with_neigh_dict=False,
        )

        kg = graph.KGInterfaceFromGraph(knowledge_graph)

    samplefunc = sample_subgraphrag.SampleSubgraphRAG(
        kg,
        t_enc,
        config.topic_pe,
        config.dde_num_rounds,
        config.dde_num_reverse_rounds,
        args.subgraph_size,
    )

    samplefunc = samplefunc.to(t_enc.device)
    print(f"Loading checkpoint at {args.retriever_path}")
    samplefunc.load_state_dict(chkpt["model_state_dict"])
    samplefunc.eval()

    llm_api = llm.LLMAPI("openai", model=args.model)

    ds = RetrieverDatasetEval(args.preprocessed_path)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False)

    ds_out = []
    i = -1
    for dp in tqdm(dl):
        i += 1

        dp_out = {
            k: ds.preprocessed_data[i][k]
            for k in [
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

        query = dp_out["question"]
        answers = [kg.get_node_label(dp_out["answer_node_id"])]

        start_time = time.time()
        _, subgraph = samplefunc(*dp)
        dp_out["graphrag_subgraph"] = subgraph
        dp_out["graphrag_retrieval_seconds"] = time.time() - start_time

        for ss in [
            5,
            10,
            30,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            250,
            350,
            500,
        ]:
            prompt = prompts.tog_answer_prompt(query, subgraph[:ss], kg)
            subgraph_answer = llm_api(prompt)
            dp_out[f"graphrag_answer_{ss}"] = subgraph_answer
            dp_out[f"graphrag_em_{ss}"] = exact_match(subgraph_answer, answers)

        ds_out.append(dp_out)

        if len(ds_out) % 10 == 0 or i == len(dl) - 1:
            with open(args.output_file, "w") as f:
                json.dump(ds_out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_full_wikidata",
        action="store_true",
        help="whether to use full Wikidata, instead of WikiKG2",
    )
    parser.add_argument(
        "--wikidata_server_urls",
        type=str,
        default="server_urls_new.txt",
        help="path of txt file with server url addresses",
    )
    parser.add_argument(
        "--wikikg_dir",
        type=str,
        help="directory containing the processed wikikg2",
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        help="path of preprocessed data files",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="the output file name"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI LLM model name"
    )
    parser.add_argument(
        "--retriever_path",
        type=str,
        help="path to checkpoint of pretrained SubgraphRAG model",
    )
    parser.add_argument(
        "--subgraph_size",
        type=int,
        default=500,
        help="max number of edges in retrieved subgraph",
    )
    args = parser.parse_args()

    main(args)
