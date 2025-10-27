# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2023 Linhao Luo (Raymond)
# Copyright (c) 2024 Graph-COM

# code partially adapted from https://github.com/RManLuo/reasoning-on-graphs/blob/master/src/utils/graph_utils.py
# and https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/dataset/retriever.py

from collections import deque

import graphviz
import networkx as nx
import numpy as np


def draw(edge_ids, relation_types, graph, seed_node=None, **kwargs):
    dot = graphviz.Digraph(engine="dot")
    for (head, tail), rel in zip(edge_ids.T, relation_types):
        dot.node(
            str(head),
            label=graph.node_labels[head],
            color="red" if head == seed_node else "black",
        )
        dot.node(
            str(tail),
            label=graph.node_labels[tail],
            color="red" if tail == seed_node else "black",
        )
        dot.edge(str(head), str(tail), label=graph.relation_labels[rel])
    return dot


def bfs_with_rule_original(graph, start_node, target_rule, max_p=10):
    # code adapted from https://github.com/RManLuo/reasoning-on-graphs/blob/master/src/utils/graph_utils.py
    # sample all paths realizing the target_rule metapath, expanding from start_node
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.extend(current_path)

        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                for connect in graph[current_node][neighbor].values():
                    rel = connect["relation_id"]
                    if rel != target_rule[len(current_path)] or len(current_path) > len(
                        target_rule
                    ):
                        continue
                    queue.append((neighbor, current_path + [connect["triple_id"]]))

    return result_paths


def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    # like bfs_with_rule_original, but return also nodes on incomplete paths
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                for connect in graph[current_node][neighbor].values():
                    rel = connect["relation_id"]
                    if rel != target_rule[len(current_path)] or len(current_path) > len(
                        target_rule
                    ):
                        continue
                    queue.append(
                        (neighbor, current_path + [(current_node, rel, neighbor)])
                    )
                    result_paths.append(connect["triple_id"])

    return result_paths


def compute_seed_answer_sp_paths(graph_triples, seeds, answers, n_rels, graph=None):
    # for each seed-answer pair, compute all shortest paths in provided set of triples
    # adapted from https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/dataset/retriever.py
    num_triples = len(graph_triples)
    if graph:
        nx_g = graph
    else:
        nx_g = nx.MultiGraph()
        for i in range(num_triples):
            h_i = graph_triples[i][0]
            r_i = graph_triples[i][1]
            t_i = graph_triples[i][2]
            nx_g.add_edge(h_i, t_i, relation_id=r_i, triple_id=i)

    path_list_ = dict()
    for q_entity_id in seeds:
        for a_entity_id in answers:
            try:
                forward_paths = list(
                    nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id)
                )
            except:
                forward_paths = []
            path_list_[(q_entity_id, a_entity_id)] = forward_paths

    # Each processed path is a list of triple IDs.
    path_list = dict()
    scores_sp = np.zeros(num_triples)
    for (q, a), qa_path_list in path_list_.items():
        path_list[(q, a)] = []
        for path in qa_path_list:
            num_triples_path = len(path) - 1
            triple_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i + 1]
                new_triple_path = []
                for connect in nx_g[h_id_i][t_id_i].values():
                    triple_id_i = connect["triple_id"]
                    hi, ri, ti = graph_triples[triple_id_i]
                    scores_sp[triple_id_i] = 1.0
                    assert ri == connect["relation_id"]
                    if len(triple_path) == 0:
                        new_triple_path.append(
                            [
                                [
                                    h_id_i,
                                    ri + n_rels * (hi != h_id_i),
                                    t_id_i,
                                ]
                            ]
                        )
                    else:
                        for p in triple_path:
                            new_triple_path.append(
                                p
                                + [
                                    [
                                        h_id_i,
                                        ri
                                        + n_rels
                                        * (
                                            hi != h_id_i
                                        ),  # if the edge on the shortest path needs to be followed in the opposite sense, we denote this by increasing the relation ID by n_rels
                                        t_id_i,
                                    ]
                                ]
                            )
                triple_path = new_triple_path

            path_list[(q, a)].extend(triple_path)
    return scores_sp, path_list
