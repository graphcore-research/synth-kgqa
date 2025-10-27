# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Liyi Chen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.


import numpy as np
from tqdm import tqdm


class Graph:
    def __init__(
        self,
        edge_ids: np.ndarray,
        relation_types: np.ndarray,
        node_labels: np.ndarray,
        relation_labels: np.ndarray,
    ):
        self.edge_ids = edge_ids.T
        self.relation_types = relation_types
        self.node_labels = node_labels
        self.relation_labels = relation_labels

        self.neighbourhood_dict = self.build_neighbourhood_dict()
        self.rel_label2id = {
            l.strip().replace(" ", "").lower(): i
            for i, l in enumerate(self.relation_labels)
        }
        self.ent_label2id = {
            l.strip().replace(" ", "").lower(): i
            for i, l in enumerate(self.node_labels)
        }

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

    def rel_label2id(self, label: str):
        return self.knowledge_graph.rel_label2id.get(
            label.strip().replace(" ", "").lower(), False
        )

    def ent_label2id(self, label: str):
        return self.knowledge_graph.ent_label2id.get(
            label.strip().replace(" ", "").lower(), False
        )
