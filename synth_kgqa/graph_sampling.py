# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import os.path as osp
import pickle
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch.utils.data as torch_data
from tqdm import tqdm


class Graph:
    def __init__(
        self,
        edge_ids: np.ndarray,
        relation_types: np.ndarray,
        node_labels: np.ndarray,
        relation_labels: np.ndarray,
        node_qids: Optional[np.ndarray] = None,
        relation_pids: Optional[np.ndarray] = None,
        exclude_rels: list[int] = None,
        exclude_nodes: list[int] = [],
    ):
        msk = ~np.isin(edge_ids, np.array(exclude_nodes)).any(0)
        self.edge_ids = edge_ids[:, msk]
        self.relation_types = relation_types[msk]
        self.relation_labels = relation_labels

        if node_qids is not None:
            self.node_labels = np.array(
                [f"{lbl} ({qid})" for lbl, qid in zip(node_labels, node_qids)]
            )
        else:
            self.node_labels = node_labels

        if relation_pids is not None:
            self.relation_labels = np.array(
                [f"{lbl} ({pid})" for lbl, pid in zip(relation_labels, relation_pids)]
            )
        else:
            self.node_labels = node_labels

        self.exclude_rels = exclude_rels

        self.neighbourhood_dict = self.build_neighbourhood_dict()
        self.degree = self.get_degree_array()
        self.connected_nodes = np.where(self.degree > 0)[0]

    def save(self, file):
        with open(file, "wb") as f_out:
            pickle.dump(self, f_out)

    @property
    def num_total_nodes(self):
        return len(self.node_labels)

    @property
    def num_connected_nodes(self):
        return len(self.connected_nodes)

    @property
    def num_total_edges(self):
        return self.edge_ids.shape[1]

    @property
    def num_total_relations(self):
        return self.relation_types.max() + 1

    def build_neighbourhood_dict(self):
        neighbourhood_dict = {node: dict() for node in range(self.num_total_nodes)}
        print("Building node neighbourhood dict")
        for edge_idx, (edge, rel) in tqdm(
            enumerate(zip(self.edge_ids.T, self.relation_types)),
            total=self.num_total_edges,
            desc="Building node neighbourhood dict",
        ):
            if rel not in self.exclude_rels:
                neighbourhood_dict[edge[0]][edge[1]] = np.append(
                    neighbourhood_dict[edge[0]].get(
                        edge[1], np.array([], dtype=np.int64)
                    ),
                    edge_idx,
                )
                neighbourhood_dict[edge[1]][edge[0]] = np.append(
                    neighbourhood_dict[edge[1]].get(
                        edge[0], np.array([], dtype=np.int64)
                    ),
                    edge_idx,
                )
        return neighbourhood_dict

    def get_degree_array(self):
        degree = np.zeros(self.num_total_nodes)
        for key, val in self.neighbourhood_dict.items():
            degree[key] = len(val)
        return degree

    @classmethod
    def from_directory(
        cls,
        directory: str,
        exclude_rels: list[int],
        exclude_nodes: list[int],
    ):
        edge_ids = np.load(osp.join(directory, "edge_ids.npy"))
        relation_types = np.load(osp.join(directory, "relation_types.npy"))
        node_labels = np.load(osp.join(directory, "node_labels.npy"), allow_pickle=True)
        relation_labels = np.load(
            osp.join(directory, "relation_labels.npy"), allow_pickle=True
        )
        if osp.isfile(osp.join(directory, "node_ids.npy")):
            print("Loading QIDs")
            node_qids = np.load(osp.join(directory, "node_ids.npy"), allow_pickle=True)
        else:
            node_qids = None
        if osp.isfile(osp.join(directory, "relation_ids.npy")):
            print("Loading PIDs")
            relation_pids = np.load(
                osp.join(directory, "relation_ids.npy"), allow_pickle=True
            )
        else:
            relation_pids = None
        return cls(
            edge_ids=edge_ids,
            relation_types=relation_types,
            node_labels=node_labels,
            relation_labels=relation_labels,
            node_qids=node_qids,
            relation_pids=relation_pids,
            exclude_rels=exclude_rels,
            exclude_nodes=exclude_nodes,
        )


class GraphSamplingDataset(torch_data.Dataset):
    def __init__(
        self,
        graph: Graph,
        subgraph_sampling_method: str,
        num_nodes: int,
        num_edges: int,
        label_sep_token: str = "-",
        triple_sep_token: str = ";",
        random_seed: Optional[int] = None,
        degree_bias: bool = False,
        seed_graphs=None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.random_seed = random_seed
        self.label_sep_token = label_sep_token
        self.triple_sep_token = triple_sep_token
        self.graph = graph
        self.seed_graphs = seed_graphs

        self.rng = np.random.default_rng(seed=self.random_seed)
        self.subgraph_sampling_method = subgraph_sampling_method
        self.degree_bias = degree_bias

    def get_added_edges(self, subgraph_nodes: np.ndarray, node: int) -> np.ndarray:
        assert self.graph is not None
        edge_idx = np.nonzero(
            (node == self.graph.edge_ids[0])
            & np.isin(self.graph.edge_ids[1], subgraph_nodes)
            | (node == self.graph.edge_ids[1])
            & np.isin(self.graph.edge_ids[0], subgraph_nodes)
        )[0]
        return edge_idx

    def sample_node(self, subgraph_nodes):
        select_nodes = subgraph_nodes
        while len(select_nodes) > 0:
            if self.degree_bias:
                exp_inv_degree = np.exp(1 / self.graph.degree[select_nodes])
                sampling_probability = exp_inv_degree / exp_inv_degree.sum()
                node = self.rng.choice(
                    select_nodes,
                    p=sampling_probability,
                )
            else:
                node = self.rng.choice(select_nodes)
            neighbours = np.fromiter(
                self.graph.neighbourhood_dict[node].keys(), dtype=np.int64
            )
            new_neighbours = neighbours[
                np.isin(neighbours, subgraph_nodes, invert=True)
            ]
            if len(new_neighbours) > 0:
                if self.degree_bias:
                    exp_inv_degree = np.exp(1 / self.graph.degree[new_neighbours])
                    sampling_probability = exp_inv_degree / exp_inv_degree.sum()
                    return self.rng.choice(
                        new_neighbours,
                        p=sampling_probability,
                    )
                else:
                    return self.rng.choice(new_neighbours)
            else:
                select_nodes = select_nodes[select_nodes != node]
        return None

    def induced_edges(self, subgraph_nodes, sampled_node):
        new_neighbourhood = np.fromiter(
            self.graph.neighbourhood_dict[sampled_node],
            dtype=np.int64,
        )
        new_neighbourhood_in_subgraph = np.isin(new_neighbourhood, subgraph_nodes)
        new_edges = np.concatenate(
            [
                self.graph.neighbourhood_dict[sampled_node][node_id]
                for node_id in new_neighbourhood[new_neighbourhood_in_subgraph]
            ]
        )
        new_neighbourhood = new_neighbourhood[np.invert(new_neighbourhood_in_subgraph)]
        return new_edges, new_neighbourhood

    def sample_graph(
        self, seed_node_index, num_nodes, num_edges
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.graph is not None
        if self.subgraph_sampling_method == "random_nodes":
            subgraph_nodes = np.sort(
                self.rng.choice(self.graph.connected_nodes, num_nodes, replace=False)
            )
            subgraph_edge_idx = np.nonzero(
                np.isin(self.graph.edge_ids[0], subgraph_nodes)
                & np.isin(self.graph.edge_ids[1], subgraph_nodes)
            )[0]
        elif self.subgraph_sampling_method == "global_neighbourhood":
            subgraph_edge_idx = np.array([], dtype=np.int64)
            subgraph_nodes = np.array([], dtype=np.int64)
            neighbourhood = np.array([], dtype=np.int64)
            while len(subgraph_nodes) < num_nodes:
                if len(neighbourhood) > 0:
                    if self.degree_bias:
                        exp_inv_degree = np.exp(1 / self.graph.degree[neighbourhood])
                        sampling_probability = exp_inv_degree / exp_inv_degree.sum()
                        sampled_node = self.rng.choice(
                            neighbourhood,
                            p=sampling_probability,
                        )
                    else:
                        sampled_node = self.rng.choice(neighbourhood)
                    neighbourhood = neighbourhood[neighbourhood != sampled_node]
                    new_edges, new_neighbourhood = self.induced_edges(
                        subgraph_nodes, sampled_node
                    )
                    subgraph_edge_idx = np.concatenate([subgraph_edge_idx, new_edges])
                else:
                    sampled_node = seed_node_index
                    new_neighbourhood = np.fromiter(
                        self.graph.neighbourhood_dict[sampled_node].keys(),
                        dtype=np.int64,
                    )
                subgraph_nodes = np.append(subgraph_nodes, sampled_node)
                neighbourhood = np.concatenate([neighbourhood, new_neighbourhood])
        elif self.subgraph_sampling_method == "node_neighbourhood":
            subgraph_edge_idx = np.array([], dtype=np.int64)
            subgraph_nodes = np.array([seed_node_index], dtype=np.int64)
            while (
                len(subgraph_nodes) < num_nodes and len(subgraph_edge_idx) < num_edges
            ):
                sampled_node = self.sample_node(subgraph_nodes)
                if sampled_node is None:
                    sampled_node = self.rng.choice(self.graph.connected_nodes)
                else:
                    new_edges, _ = self.induced_edges(subgraph_nodes, sampled_node)
                    subgraph_edge_idx = np.concatenate([subgraph_edge_idx, new_edges])
                subgraph_nodes = np.append(subgraph_nodes, sampled_node)
        elif self.subgraph_sampling_method == "bfs":
            subgraph_edge_idx = np.array([], dtype=np.int64)
            subgraph_nodes = np.array([seed_node_index], dtype=np.int64)
            node_queue = list(self.graph.neighbourhood_dict[seed_node_index])
            if len(node_queue) > num_nodes:
                node_queue = random.sample(node_queue, num_nodes)
            else:
                random.shuffle(node_queue)
            while (
                len(subgraph_nodes) < num_nodes and len(subgraph_edge_idx) < num_edges
            ):
                if len(node_queue) == 0:
                    return None, None
                new_node_id = node_queue.pop(0)

                if len(node_queue) + len(subgraph_nodes) < num_nodes:
                    add_to_queue = [
                        node_id
                        for node_id in self.graph.neighbourhood_dict[new_node_id]
                        if node_id not in subgraph_nodes and node_id not in node_queue
                    ]
                    if len(add_to_queue) + len(node_queue) > num_nodes:
                        add_to_queue = random.sample(
                            add_to_queue, num_nodes - len(node_queue)
                        )
                    else:
                        random.shuffle(add_to_queue)
                    node_queue.extend(add_to_queue)
                new_edges, _ = self.induced_edges(subgraph_nodes, new_node_id)
                subgraph_edge_idx = np.concatenate([subgraph_edge_idx, new_edges])
                subgraph_nodes = np.append(subgraph_nodes, new_node_id)
        else:
            raise ValueError(f"Invalid sampling method {self.subgraph_sampling_method}")
        return np.sort(subgraph_nodes), subgraph_edge_idx

    def choose_nontrivial_neigh(self, seed):
        for neigh, edges in self.graph.neighbourhood_dict.get(seed, dict()).items():
            found = False
            for edge in edges:
                edge_type = self.graph.relation_types[edge]
                msk = np.logical_and(
                    self.graph.relation_types == edge_type,
                    (self.graph.edge_ids == neigh).any(0),
                )
                if msk.sum() > 1:
                    found = True
                    break
            if found:
                return neigh, edge
        return None

    def expand_graph(self, seed_graph, num_nodes, num_edges):
        kg_triples = np.stack(
            [
                self.graph.edge_ids[0, :],
                self.graph.relation_types,
                self.graph.edge_ids[1, :],
            ]
        ).T
        ans_subg = seed_graph["answer_subgraph"]
        row_idx, subgraph_edge_idx = np.where(
            (np.array(ans_subg)[:, None] == kg_triples[None, :]).all(-1)
        )
        assert len(row_idx) == len(ans_subg)

        all_nodes = np.unique(np.concatenate(np.array(ans_subg)[:, [0, 2]])).tolist()
        nodes_with_neigh = []
        for node in all_nodes:
            neighs = self.graph.neighbourhood_dict.get(node, dict())
            if len(neighs.keys()) > 0:
                nodes_with_neigh.append(node)
        if len(nodes_with_neigh) == 0:
            return np.sort(np.array(all_nodes)), subgraph_edge_idx

        branch_node = self.rng.choice(nodes_with_neigh)
        forced = False
        for t in ans_subg:
            if seed_graph["seed_nodes_id"][0] == t[0]:
                if t[2] in nodes_with_neigh:
                    branch_node = t[2]
                    forced = True
                break
            elif seed_graph["seed_nodes_id"][0] == t[2]:
                if t[0] in nodes_with_neigh:
                    branch_node = t[0]
                    forced = True
                break

        redo = True
        n_loops = 0
        while redo and n_loops < 50:
            n_loops += 1
            nontrivial_neigh = self.choose_nontrivial_neigh(branch_node)
            if nontrivial_neigh:
                new_edge_id = nontrivial_neigh[1]
            else:
                neighs = self.graph.neighbourhood_dict[branch_node]
                new_edge_id = self.rng.choice(
                    neighs[np.random.choice(list(neighs.keys()))]
                )
            h, t = self.graph.edge_ids[:, new_edge_id].tolist()
            if h not in all_nodes:
                all_nodes.append(h)
                redo = False
            if t not in all_nodes:
                all_nodes.append(t)
                redo = False

            if not redo:
                subgraph_edge_idx = np.concatenate([subgraph_edge_idx, [new_edge_id]])

            if (not forced) or n_loops > 25:
                branch_node = self.rng.choice(nodes_with_neigh)

        return np.sort(np.array(all_nodes)), subgraph_edge_idx

    def expand_graph_from_answer(self, seed_graph, num_nodes, num_edges):
        kg_triples = np.stack(
            [
                self.graph.edge_ids[0, :],
                self.graph.relation_types,
                self.graph.edge_ids[1, :],
            ]
        ).T
        ans_subg = seed_graph["answer_subgraph"]
        row_idx, subgraph_edge_idx = np.where(
            (np.array(ans_subg)[:, None] == kg_triples[None, :]).all(-1)
        )
        assert len(row_idx) == len(ans_subg)

        all_nodes = np.unique(np.concatenate(np.array(ans_subg)[:, [0, 2]])).tolist()
        nodes_with_neigh = []
        for node in all_nodes:
            neighs = self.graph.neighbourhood_dict.get(node, dict())
            if len(neighs.keys()) > 0:
                nodes_with_neigh.append(node)
        if len(nodes_with_neigh) == 0:
            return np.sort(np.array(all_nodes)), subgraph_edge_idx

        branch_node = self.rng.choice(nodes_with_neigh)
        forced = False
        if seed_graph["answer_node_id"] in nodes_with_neigh:
            branch_node = seed_graph["answer_node_id"]
            forced = True

        redo = True
        n_loops = 0
        while redo and n_loops < 50:
            n_loops += 1
            nontrivial_neigh = self.choose_nontrivial_neigh(branch_node)
            if nontrivial_neigh:
                new_edge_id = nontrivial_neigh[1]
            else:
                neighs = self.graph.neighbourhood_dict[branch_node]
                new_edge_id = self.rng.choice(
                    neighs[np.random.choice(list(neighs.keys()))]
                )
            h, t = self.graph.edge_ids[:, new_edge_id].tolist()
            if h not in all_nodes:
                all_nodes.append(h)
                redo = False
            if t not in all_nodes:
                all_nodes.append(t)
                redo = False

            if not redo:
                subgraph_edge_idx = np.concatenate([subgraph_edge_idx, [new_edge_id]])

            if (not forced) or n_loops > 25:
                branch_node = self.rng.choice(nodes_with_neigh)

        return np.sort(np.array(all_nodes)), subgraph_edge_idx

    def flatten(self, edge_ids, relation_types):
        """returns a textualised version of the subgraph as set of triples
        `head_label0 <sep0> relation_label0 <sep0> tail_label0 <sep1> head_label1 <sep0> relation_label1 <sep0> tail_label1 ...`
        """
        return self.triple_sep_token.join(
            [
                f"{self.graph.node_labels[edge[0]]}{self.label_sep_token}"
                f"{self.graph.relation_labels[rel]}{self.label_sep_token}"
                f"{self.graph.node_labels[edge[1]]}"
                for edge, rel in zip(edge_ids.T, relation_types)
            ]
        )

    def __getitem__(self, index) -> Dict:
        if self.subgraph_sampling_method == "expand":
            assert len(self.seed_graphs) > 0
            seed_graph = self.seed_graphs[index]
            node_ids, edge_idx = self.expand_graph(
                seed_graph, self.num_nodes, self.num_edges
            )
        elif self.subgraph_sampling_method == "expand_answer":
            assert len(self.seed_graphs) > 0
            seed_graph = self.seed_graphs[index]
            node_ids, edge_idx = self.expand_graph_from_answer(
                seed_graph, self.num_nodes, self.num_edges
            )
        else:
            seed_node = self.graph.connected_nodes[index].item()
            # Sample nodes and identify index of induced edges
            node_ids, edge_idx = self.sample_graph(
                seed_node, self.num_nodes, self.num_edges
            )
        if node_ids is None:
            return None

        # randomly sample edges if the induced edges are more than
        # num_total_edges
        if edge_idx.shape[0] > self.num_edges:
            edge_idx = np.sort(self.rng.choice(edge_idx, self.num_edges, replace=False))

        # generate edges, relation types and labels of induced subgraph
        edge_ids = self.graph.edge_ids[:, edge_idx]
        relation_types = self.graph.relation_types[edge_idx]

        out = dict(
            edge_ids=edge_ids,
            relation_types=relation_types,
            seed_node=(
                seed_node
                if "expand" not in self.subgraph_sampling_method
                else seed_graph["answer_subgraph"]
            ),
            flattened_graph=self.flatten(
                edge_ids,
                relation_types,
            ),
            num_nodes=len(node_ids),
            num_triples=len(edge_idx),
        )
        if "expand" in self.subgraph_sampling_method:
            out.update(dict(expansion_source=seed_graph["source"]))
        return out

    def __len__(self):
        return self.graph.num_connected_nodes
