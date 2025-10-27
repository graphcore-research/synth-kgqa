# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import numpy as np
from tqdm import tqdm

from llm_graph_walk.client_light import MultiServerWikidataQueryClient


class Graph:
    def __init__(
        self,
        edge_ids: np.ndarray,
        relation_types: np.ndarray,
        node_labels: np.ndarray,
        relation_labels: np.ndarray,
        with_neigh_dict: bool = True,
    ):
        self.edge_ids = edge_ids.T
        self.relation_types = relation_types
        self.node_labels = node_labels
        self.relation_labels = relation_labels

        if with_neigh_dict:
            self.neighbourhood_dict = self.build_neighbourhood_dict()
            self.degree = self.get_degree_array()

    @property
    def num_total_nodes(self):
        return len(self.node_labels)

    @property
    def num_total_edges(self):
        return self.edge_ids.shape[0]

    @property
    def num_total_relations(self):
        return len(self.relation_labels)

    def build_neighbourhood_dict(self):
        neighbourhood_dict = {
            node: {"head": dict(), "tail": dict()}
            for node in range(self.num_total_nodes)
        }
        for idx, (edge, rel) in tqdm(
            enumerate(zip(self.edge_ids, self.relation_types)),
            total=self.num_total_edges,
            desc="Building node neighbourhood dict",
        ):
            h_id = int(edge[0])
            t_id = int(edge[1])
            r_id = int(rel)
            if r_id in neighbourhood_dict[h_id]["head"]:
                neighbourhood_dict[h_id]["head"][r_id].append(t_id)
            else:
                neighbourhood_dict[h_id]["head"][r_id] = [t_id]

            if r_id in neighbourhood_dict[t_id]["tail"]:
                neighbourhood_dict[t_id]["tail"][r_id].append(h_id)
            else:
                neighbourhood_dict[t_id]["tail"][r_id] = [h_id]
        return neighbourhood_dict

    def get_degree_array(self):
        degree = np.zeros(self.num_total_nodes)
        for head, rel_dict in self.neighbourhood_dict.items():
            degree[head] = sum([len(tails) for tails in rel_dict.values()])
        return degree


class KGInterface:
    def get_node_label(self, node_id: int | str):
        raise NotImplementedError()

    def get_relation_label(self, relation_id: int | str):
        raise NotImplementedError()

    def get_entity_relations(self, node_id: int | str):
        raise NotImplementedError()


class KGInterfaceFromGraph(KGInterface):
    def __init__(
        self,
        kg: Graph,
    ):
        self.knowledge_graph = kg

    def get_node_label(self, node_id: int):
        return self.knowledge_graph.node_labels[node_id]

    def get_relation_label(self, relation_id: int):
        return self.knowledge_graph.relation_labels[relation_id]

    def get_entity_relations(self, node_id: int):
        return self.knowledge_graph.neighbourhood_dict[node_id]


class KGInterfaceFromWikidata(KGInterface):
    def __init__(
        self,
        server_address_list: str = "server_urls_new.txt",
    ):
        # wiki_client
        with open(server_address_list, "r") as f:
            server_addrs = f.readlines()
            server_addrs = [addr.strip() for addr in server_addrs]
        self.wiki_client = MultiServerWikidataQueryClient(server_addrs)
        self.wiki_client.test_connections()

    def get_node_label(self, node_id: str):
        query_res = list(self.wiki_client.query_all("qid2label", node_id))
        return (
            query_res[0] if len(query_res) > 0 else node_id
        )  # if node_id is not a qid, it's treated as a wikidata value and thus returned

    def get_relation_label(self, relation_id: str):
        return list(self.wiki_client.query_all("pid2label", relation_id))[0]

    def get_entity_relations(self, node_id: str):
        return self.wiki_client.query_all("get_all_relations_of_an_entity", node_id)
