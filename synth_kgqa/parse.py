# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import argparse


def parse_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", type=str)
    parser.add_argument(
        "--subgraph-sampling-method",
        type=str,
        choices=[
            "bfs",
            "node_neighbourhood",
            "global_neighbourhood",
            "expand",
            "expand_answer",
        ],
        default="node_neighbourhood",
        help="sampling scheme for seed subgraph Q",
    )
    parser.add_argument(
        "--num-subgraph-nodes",
        type=int,
        default=10,
        help="max number of nodes in the seed subgraph Q",
    )
    parser.add_argument(
        "--num-subgraph-edges",
        type=int,
        default=30,
        help="max number of edges in the seed subgraph Q",
    )
    parser.add_argument("--label-sep-token", type=str, default="-")
    parser.add_argument("--triple-sep-token", type=str, default=";")
    parser.add_argument("--random-seed", type=int, default=1111)
    parser.add_argument(
        "--degree-bias",
        action="store_true",
        help="in the construction of Q, sample neighbors with probability proportional to the inverse of their degree",
    )
    parser.add_argument("--api-name", type=str, default="openai")
    parser.add_argument("--llm", type=str, default="gpt-4.1")
    parser.add_argument(
        "--exclude-rels",
        type=int,
        nargs="*",
        default=[
            245,
            269,
            349,
            427,
        ],  #'sex or gender', 'instance of', 'diplomatic relation', 'given name' for wikikg2
        help="list of relation types to exclude from the KG",
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        default=None,
        help="path to np array containing entities to exclude from the KG",
    )
    parser.add_argument(
        "--num-edges",
        type=int,
        default=2,
        help="number k of edges in the answer subgraph",
    )
    parser.add_argument(
        "--num-samples", type=int, help="number of questions to generate"
    )
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--seed-path", type=str, default=None)
    parser.add_argument("--graph-to-expand", type=str, default=None)
    parser.add_argument("--max-n-tries", type=int, default=8)
    return parser.parse_args()


def parse_processing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-name", type=str, default="openai")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--kg-path", type=str)
    parser.add_argument("--qa-path", type=str)
    parser.add_argument(
        "--max-num-answers",
        type=int,
        default=10,
        help="discard questions that have more than this number of answers in Wikidata",
    )
    return parser.parse_args()
