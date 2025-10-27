# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import json
import os.path as osp
import pickle

import dgl
import networkx as nx
import numpy as np
from kg_utils import bfs_with_rule, bfs_with_rule_original, compute_seed_answer_sp_paths
from tqdm import tqdm

N_NODES = 2500
MIN_SCORE = 1e-6


def get_in_out_neigh_edges(graph, seeds, k):
    k_hop_neigh_out = (
        dgl.khop_out_subgraph(
            graph,
            nodes=seeds,
            k=k,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    k_hop_neigh_in = (
        dgl.khop_in_subgraph(
            graph,
            nodes=seeds,
            k=k,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    one_hop_neigh_out = (
        dgl.khop_out_subgraph(
            graph,
            nodes=seeds,
            k=1,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    one_hop_neigh_in = (
        dgl.khop_in_subgraph(
            graph,
            nodes=seeds,
            k=1,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    k_hop_unique = np.unique(np.concatenate([k_hop_neigh_out, k_hop_neigh_in]))
    out_in = (
        dgl.khop_in_subgraph(
            graph,
            nodes=np.unique(edge_ids[:, one_hop_neigh_out].flatten()),
            k=1,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    if len(out_in) > 60000:
        out_in = np.random.choice(out_in, size=(60000,), replace=False)
    in_out = (
        dgl.khop_out_subgraph(
            graph,
            nodes=np.unique(edge_ids[:, one_hop_neigh_in].flatten()),
            k=1,
            store_ids=True,
            relabel_nodes=False,
        )
        .edata[dgl.EID]
        .numpy()
    )
    if len(in_out) > 60000:
        in_out = np.random.choice(in_out, size=(60000,), replace=False)
    return np.unique(np.concatenate([k_hop_unique, out_in, in_out]))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--kg-path", type=str)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    edge_ids = np.load(osp.join(args.kg_path, "edge_ids.npy"))
    relation_types = np.load(osp.join(args.kg_path, "relation_types.npy"))
    node_labels = np.load(osp.join(args.kg_path, "node_labels.npy"), allow_pickle=True)
    relation_labels = np.load(
        osp.join(args.kg_path, "relation_labels.npy"), allow_pickle=True
    )
    node_ids = np.load(osp.join(args.kg_path, "node_ids.npy"), allow_pickle=True)
    relation_ids = np.load(
        osp.join(args.kg_path, "relation_ids.npy"),
        allow_pickle=True,
    )
    rev_lkp_ent = {ent: idx for idx, ent in enumerate([int(x[1:]) for x in node_ids])}
    rev_lkp_rels = {
        ent: idx for idx, ent in enumerate([int(x[1:]) for x in relation_ids])
    }
    full_triples = np.stack([edge_ids[0], relation_types, edge_ids[1]]).T

    # Full KG
    graph = dgl.graph((edge_ids[0], edge_ids[1]), num_nodes=len(node_labels))

    train_gt = json.load(open(args.dataset, "rb"))
    out_file = f"{args.dataset}_scores_{args.block}{args.suffix}.pkl"

    # process only a portion of the dataset, for parallel computing
    span = list(range(len(train_gt)))
    if args.block_size:
        span = span[args.block * args.block_size : (args.block + 1) * args.block_size]

    train_sp = []
    it = -1
    for i in tqdm(span):
        if it % 50 == 0:
            pickle.dump(train_sp, open(out_file, "wb"))
        it += 1
        sample = train_gt[i]
        seed_nodes = sample["seed_nodes_id"]
        if len(seed_nodes) == 0:
            raise ValueError
        # Wikikg2 IDs of all answer nodes
        all_answ_nodes = [
            rev_lkp_ent.get(int(qid[1:]), -1)
            for qid in sample["all_answers"]
            if qid[0] == "Q"
        ]
        all_answ_nodes = [x for x in all_answ_nodes if x > -1]
        answ_nodes = [sample["answer_node_id"]]
        if len(answ_nodes) == 0:
            raise ValueError
        sample["topic_entity_id"] = seed_nodes
        sample["answer_id"] = answ_nodes

        gt_subg = np.array(sample["answer_subgraph"])
        # WikiKG2 triples of full answer subgraph
        gt_subg_full = np.array(
            [
                [
                    rev_lkp_ent.get(int(t[0][1:]), -1),
                    rev_lkp_rels.get(int(t[1][1:]), -1),
                    rev_lkp_ent.get(int(t[2][1:]), -1),
                ]
                for t in sample["full_subgraph"]
            ]
        )
        gt_subg_full = gt_subg_full[(gt_subg_full >= 0).all(-1), :]

        # sample k-hop neighbourood
        neigh = get_in_out_neigh_edges(graph, seed_nodes, k=max(3, sample["n_hops"]))

        # ground truth seed -> answer paths
        sample["seed_answer_paths"] = compute_seed_answer_sp_paths(
            gt_subg.tolist(), seed_nodes, answ_nodes, len(relation_labels)
        )[1]
        # ground truth metapaths (undirected)
        rules_gt = {}
        unique_gt_rels = []
        for (seed, _), path_list in sample["seed_answer_paths"].items():
            rules_gt[seed] = []
            for p in path_list:
                rule = [t[1] % len(relation_labels) for t in p]
                unique_gt_rels.extend(rule)
                if rule not in rules_gt[seed]:
                    rules_gt[seed].append(rule)
        unique_gt_rels = np.unique(unique_gt_rels)
        triples_sel = np.where(np.isin(relation_types, unique_gt_rels))[0]
        # sample all triples on ground truth metapaths from seed nodes
        nx_g = nx.MultiGraph()
        for i, k in enumerate(triples_sel):
            nx_g.add_edge(
                edge_ids[0, k].item(),
                edge_ids[1, k].item(),
                relation_id=relation_types[k].item(),
                triple_id=i,
            )
        triples_on_gt_rules = []
        for i, (seed, rule_list) in enumerate(rules_gt.items()):
            per_rule_sp = []
            for k, rule in enumerate(rule_list):
                orig = bfs_with_rule_original(nx_g, seed, rule)[:650]
                if len(orig) < 650:
                    orig += bfs_with_rule(nx_g, seed, rule)[: (650 - len(orig))]
                per_rule_sp.extend(orig)
            triples_on_gt_rules.extend(per_rule_sp)

        # Update neighborood
        neigh = np.unique(np.concatenate([neigh, triples_sel[triples_on_gt_rules]]))
        neigh_triples = np.stack(
            [edge_ids[0, neigh], relation_types[neigh], edge_ids[1, neigh]]
        ).T.tolist()

        # sample all seed -> answer shortest paths
        nx_g = nx.MultiGraph()
        for i, k in enumerate(neigh):
            nx_g.add_edge(
                edge_ids[0, k].item(),
                edge_ids[1, k].item(),
                relation_id=relation_types[k].item(),
                triple_id=i,
            )
        seed_ans_sp_paths = compute_seed_answer_sp_paths(
            neigh_triples, seed_nodes, answ_nodes, len(relation_labels), graph=nx_g
        )
        sample["seed_answer_paths_sp"] = seed_ans_sp_paths[1]
        # and sp metapaths
        rules_sp = {}
        for (seed, _), path_list in sample["seed_answer_paths_sp"].items():
            rules_sp[seed] = []
            for p in path_list:
                rule = [t[1] % len(relation_labels) for t in p]
                if rule not in rules_sp[seed]:
                    rules_sp[seed].append(rule)

        # triples to re-add in final graph
        triples_on_gt_converted = np.where(
            np.isin(neigh, triples_sel[triples_on_gt_rules])
        )[0].tolist()
        triples_on_sp_rules = []
        for i, (seed, rule_list) in enumerate(rules_sp.items()):
            max_per_seed = (7500 - len(triples_on_sp_rules)) // (len(rules_sp) - i)
            per_rule_sp = []
            for k, rule in enumerate(rule_list):
                max_per_rule = (max_per_seed - len(per_rule_sp)) // (len(rule_list) - k)
                per_rule_sp.extend(bfs_with_rule_original(nx_g, seed, rule))
                per_rule_sp = np.unique(per_rule_sp).tolist()
            triples_on_sp_rules.extend(per_rule_sp)
            triples_on_sp_rules = np.unique(triples_on_sp_rules).tolist()
        if len(triples_on_sp_rules) > 7500:
            triples_on_sp_rules = np.random.choice(
                triples_on_sp_rules, (7500,)
            ).tolist()
        triples_on_sp_rules = np.unique(
            triples_on_sp_rules + triples_on_gt_converted
        ).tolist()

        # personalized pagerank to reduce size of neighborood
        personalization = {s: 1.0 / len(seed_nodes) for s in seed_nodes}
        ppr = nx.pagerank(nx_g, personalization=personalization)
        ppr_values = np.fromiter(ppr.values(), dtype=np.float32)
        top_ids = np.argsort(ppr_values)[-N_NODES:][::-1]
        top_ids = top_ids[ppr_values[top_ids] > MIN_SCORE]
        top_entities = np.fromiter(ppr.keys(), dtype=np.int32)[top_ids]

        # select nodes for final graph, based on PPR scores
        nodes_in_gt = np.unique(gt_subg_full[:, [0, 2]])
        to_keep_always = np.isin(edge_ids[:, neigh], nodes_in_gt).any(0)
        not_essential = np.logical_and(
            to_keep_always, np.invert(np.isin(edge_ids[:, neigh], nodes_in_gt).all(0))
        )
        if to_keep_always.sum() > 7500:
            to_keep_always[np.where(not_essential)[0][7500:]] = False
        msk = np.logical_or(
            np.isin(edge_ids[:, neigh], top_entities).all(0), to_keep_always
        )
        num_triples_max = 20000
        while msk.sum() > num_triples_max:
            top_entities = top_entities[:-100]
            msk = np.logical_or(
                np.isin(edge_ids[:, neigh], top_entities).all(0), to_keep_always
            )
            if len(top_entities) < 150:
                to_keep_pos = np.where(msk)[0]
                to_keep_pos = np.random.choice(
                    to_keep_pos, size=(num_triples_max - 2,), replace=False
                )
                msk = np.array([False] * len(msk))
                msk[to_keep_pos] = True

        # re-add triples to always keep
        msk[triples_on_sp_rules] = True
        new_edges = neigh[msk]
        assert len(new_edges) > 0
        triples = np.stack(
            [
                edge_ids[0, new_edges],
                relation_types[new_edges],
                edge_ids[1, new_edges],
            ]
        ).T
        sp_scores = seed_ans_sp_paths[0][msk]

        comp = (triples[None, :] == gt_subg[:, None]).all(-1)
        missing = ~comp.any(1)
        scores = 0.0 + comp.any(0)
        to_add = gt_subg[missing]
        triples = np.concatenate([triples, to_add], axis=0)
        scores = np.concatenate([scores, np.ones(to_add.shape[0])])
        sp_scores = np.concatenate([sp_scores, np.zeros(to_add.shape[0])])
        # shuffle
        shf = np.random.permutation(triples.shape[0])
        triples = triples[shf]
        scores = scores[shf]
        sp_scores = sp_scores[shf]
        assert np.sum(scores) == gt_subg.shape[0]

        sample["h_id_list"] = triples[:, 0].tolist()
        sample["t_id_list"] = triples[:, 2].tolist()
        sample["r_id_list"] = triples[:, 1].tolist()
        sample["triple_scores"] = scores.tolist()
        sample["triple_scores_sp"] = sp_scores.tolist()

        train_sp.append(sample)

    pickle.dump(train_sp, open(out_file, "wb"))
    print(f"{out_file} FINISHED!")
