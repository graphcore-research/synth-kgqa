# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import itertools
import re
import time
from string import ascii_lowercase

import networkx as nx
import numpy as np

ISO_TEMPLATES = {
    "(1)": [["seed", "ans"]],
    "(2)": [["seed", "a"], ["a", "ans"]],
    "(1)(1)": [["seed1", "ans"], ["ans", "seed2"]],
    "(2)(1)": [["seed1", "a"], ["a", "ans"], ["seed2", "ans"]],
    "((1)(1))": [["seed1", "a"], ["seed2", "a"], ["a", "ans"]],
    "(3)": [["seed", "a"], ["a", "b"], ["b", "ans"]],
    "(2)(2)": [["seed1", "a"], ["a", "ans"], ["seed2", "b"], ["b", "ans"]],
    "(3)(1)": [["seed1", "a"], ["a", "b"], ["b", "ans"], ["seed2", "ans"]],
    "((2)(1))": [["seed1", "a"], ["a", "b"], ["b", "ans"], ["seed2", "b"]],
    "((1)(1))(1)": [["seed1", "a"], ["seed2", "a"], ["a", "ans"], ["ans", "seed3"]],
    "(2(1)(1))": [["seed1", "a"], ["seed2", "a"], ["a", "b"], ["b", "ans"]],
    "(1)(1)(1)": [["seed1", "ans"], ["seed2", "ans"], ["ans", "seed3"]],
    "((1)(1)(1))": [["seed1", "a"], ["seed2", "a"], ["seed3", "a"], ["ans", "a"]],
    "(2)(2)(1)": [
        ["seed1", "a"],
        ["seed2", "b"],
        ["seed3", "ans"],
        ["ans", "a"],
        ["ans", "b"],
    ],
    "((3)(1))": [
        ["seed1", "a"],
        ["seed2", "b"],
        ["a", "ans"],
        ["b", "c"],
        ["c", "a"],
    ],
    "(4)(1)": [
        ["seed1", "ans"],
        ["seed2", "a"],
        ["a", "b"],
        ["b", "c"],
        ["c", "ans"],
    ],
    "(4)": [["seed", "a"], ["a", "b"], ["b", "c"], ["c", "ans"]],
    "((1)(1))(2)": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["a", "ans"],
        ["b", "ans"],
        ["b", "seed3"],
    ],
    "(2)(1)(1)": [["seed1", "ans"], ["seed2", "ans"], ["ans", "a"], ["a", "seed3"]],
    "((2)(1)(1))": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["ans", "a"],
        ["a", "b"],
        ["b", "seed3"],
    ],
    "((3)(1)(1))": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["ans", "a"],
        ["a", "b"],
        ["b", "c"],
        ["c", "seed3"],
    ],
    "((1)(1))(1)(1)": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["ans", "a"],
        ["ans", "seed3"],
        ["ans", "seed4"],
    ],
    "((2)(1))(1)": [
        ["seed1", "a"],
        ["seed2", "b"],
        ["b", "a"],
        ["ans", "a"],
        ["ans", "seed3"],
    ],
    "((3)(1))(1)": [
        ["seed1", "a"],
        ["seed2", "c"],
        ["c", "b"],
        ["b", "a"],
        ["ans", "a"],
        ["ans", "seed3"],
    ],
    "((1)(1)(1))(1)": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["seed3", "a"],
        ["ans", "a"],
        ["ans", "seed4"],
    ],
    "((1)(1)(1))(2)": [
        ["seed1", "a"],
        ["seed2", "a"],
        ["seed3", "a"],
        ["ans", "a"],
        ["ans", "b"],
        ["b", "seed4"],
    ],
    "(1)(1)(1)(1)": [
        ["seed1", "ans"],
        ["seed2", "ans"],
        ["seed3", "ans"],
        ["seed4", "ans"],
    ],
    "(1)(1)(1)(1)(1)": [
        ["seed1", "ans"],
        ["seed2", "ans"],
        ["seed3", "ans"],
        ["seed4", "ans"],
        ["seed5", "ans"],
    ],
    "(5)": [["seed", "a"], ["a", "b"], ["b", "c"], ["c", "d"], ["d", "ans"]],
}

# GRAPH_SIZE_TO_TEMPLATE[n_edges][n_seeds] is the set of acceptable graph isomorphism classes
# for graphs with n_edges edges and n_seeds seed nodes
# (a single answer node, different from seeds, is always assumed)
GRAPH_SIZE_TO_TEMPLATE = {
    1: {1: ["(1)"]},
    2: {1: ["(2)"], 2: ["(1)(1)"]},
    3: {1: ["(3)"], 2: ["(2)(1)", "((1)(1))"], 3: ["(1)(1)(1)"]},
    4: {
        1: ["(4)"],
        2: ["(2)(2)", "(3)(1)", "((2)(1))", "(2(1)(1))"],
        3: ["((1)(1))(1)", "((1)(1)(1))", "(2)(1)(1)"],
        4: ["(1)(1)(1)(1)"],
    },
    5: {
        1: ["(5)"],
        2: ["((3)(1))", "(4)(1)"],
        3: ["(2)(2)(1)", "((1)(1))(2)", "((2)(1)(1))", "((2)(1))(1)"],
        4: ["((1)(1))(1)(1)", "((1)(1)(1))(1)"],
        5: ["(1)(1)(1)(1)(1)"],
    },
    6: {3: ["((3)(1)(1))", "((3)(1))(1)"], 4: ["((1)(1)(1))(2)"]},
}


def build_n_hops_mapper():
    n_hops_mapper = dict()
    for k, struct in ISO_TEMPLATES.items():
        graph = nx.Graph()
        for edge in struct:
            graph.add_edge(edge[0], edge[1])
        n_hops_mapper[k] = nx.eccentricity(graph, "ans")
    return n_hops_mapper


def compare_with_template(edge_list, seed_nodes, ans_node, reference_name):
    def node_match(x, y):
        return x["color"] == y["color"]

    graph = nx.Graph()
    for edge in edge_list:
        graph.add_edge(edge[0], edge[2])
    for n in graph.nodes:
        if n in seed_nodes:
            graph.nodes[n]["color"] = "red"
        elif n == ans_node:
            graph.nodes[n]["color"] = "green"
        else:
            # intermediate nodes
            graph.nodes[n]["color"] = "white"

    ref = nx.Graph()
    for edge in ISO_TEMPLATES[reference_name]:
        ref.add_edge(edge[0], edge[1])
    for n in ref.nodes:
        if "seed" in n:
            ref.nodes[n]["color"] = "red"
        elif "ans" in n:
            ref.nodes[n]["color"] = "green"
        else:
            ref.nodes[n]["color"] = "white"

    return nx.is_isomorphic(graph, ref, node_match=node_match)


def check_template(subgraph, seed, ans):
    if len(subgraph) in GRAPH_SIZE_TO_TEMPLATE.keys():
        if len(seed) in GRAPH_SIZE_TO_TEMPLATE[len(subgraph)].keys():
            for template in GRAPH_SIZE_TO_TEMPLATE[len(subgraph)][len(seed)]:
                if compare_with_template(subgraph, seed, ans, template):
                    return template
    return False


def parse_llm_label(string, node_labels, relation_labels, qid_lookup, pid_lookup):
    # parse label and QID/PID from LLM output and check they correspond
    string_parse = re.search(r"(.*) \(((P|Q)[0-9]*)\)", string)
    if not string_parse or len(string_parse.groups()) < 2:
        return []
    label, pqid = string_parse.group(1), string_parse.group(2)
    if pqid[0] == "Q":
        wiki_id = qid_lookup.get(pqid, -1)
        labs = node_labels
    else:
        wiki_id = pid_lookup.get(pqid, -1)
        labs = relation_labels
    if (
        wiki_id < 0
        or labs[wiki_id].replace(" ", "").lower() != label.replace(" ", "").lower()
    ):
        return []
    else:
        return label, pqid, wiki_id


def check_redundant_subquery(
    sparql,
    answer_subgraph,
    answer_node,
    all_answers,
    seed_nodes,
    node_qids,
    relation_pids,
):
    intermediate_labels = [f"?{i}" for i in ascii_lowercase]
    graph = nx.Graph()
    i = 0
    for edge in answer_subgraph:
        graph.add_edge(edge[0], edge[2], triple_id=i)
        i += 1
    k = 0
    for n in graph.nodes:
        if n in seed_nodes:
            graph.nodes[n]["label"] = f"wd:{node_qids[n]}"
        elif n == answer_node:
            graph.nodes[n]["label"] = "?answer"
        else:
            graph.nodes[n]["label"] = intermediate_labels[k]
            k += 1
    sh_paths = {}
    for s in seed_nodes:
        sh_path_seed_ans = list(
            nx.shortest_paths.all_shortest_paths(source=s, target=answer_node, G=graph)
        )[0]
        query = ""
        edges = []
        for i in range(len(sh_path_seed_ans) - 1):
            edge = graph[sh_path_seed_ans[i]][sh_path_seed_ans[i + 1]]["triple_id"]
            h, r, t = answer_subgraph[edge]
            edges.append([h, r, t])
            query += f"{graph.nodes[h]['label']} wdt:{relation_pids[r]} {graph.nodes[t]['label']}. "
        sh_paths[s] = {"sh_path": edges, "query": query}

    sufficient_paths = dict()
    for i in range(len(seed_nodes)):
        for seeds in itertools.combinations(seed_nodes, i + 1):
            query = "SELECT ?answer WHERE { "
            subg = []
            for s in seeds:
                query += sh_paths[s]["query"]
                subg.extend(sh_paths[s]["sh_path"])
            query += "}"
            n_tries = 0
            while n_tries < 3:
                try:
                    time.sleep(0.0002)  # to avoid exceeding Wikidata query limits
                    sparql.setQuery(query)
                    ret = sparql.queryAndConvert()
                    extended_answ = [
                        result["answer"]["value"].split("/")[-1]
                        for result in ret["results"]["bindings"]
                    ]
                    if len(np.unique(extended_answ)) <= len(all_answers):
                        sufficient_paths[seeds] = {
                            "graph": np.unique(np.array(subg), axis=0).tolist(),
                            "query": query,
                        }
                    break
                except Exception as e:
                    n_tries += 1

    if len(sufficient_paths.keys()) == 0:
        return None, None, None

    minimal_subgraphs = dict()
    minimal_queries = dict()
    for s in seed_nodes:
        minimal_subg = []
        for seeds, suff_p in sufficient_paths.items():
            if s in seeds:
                minimal_subg.append(suff_p)
        minimal_subg_lens = min([len(p["graph"]) for p in minimal_subg])
        minimal_subgraphs[s] = [
            x["graph"] for x in minimal_subg if len(x["graph"]) == minimal_subg_lens
        ]
        minimal_queries[s] = [
            x["query"] for x in minimal_subg if len(x["graph"]) == minimal_subg_lens
        ]

    min_len_path = 9999
    for seeds, path in sufficient_paths.items():
        if len(path["graph"]) < min_len_path:
            effective_seeds = seeds
            effective_graph = path["graph"]
            min_len_path = len(effective_graph)

    minimal_template = check_template(effective_graph, effective_seeds, answer_node)

    return minimal_template, minimal_subgraphs, minimal_queries
