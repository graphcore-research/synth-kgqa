# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Graph-COM

# code adapted from https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/model/retriever.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from llm_graph_walk import graph, text_encoder


class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())

        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)

        return result_list


class SampleSubgraphRAG(nn.Module):
    def __init__(
        self,
        kg_interface: graph.KGInterface,
        text_encoder: text_encoder.TextEncoder,
        topic_pe=True,
        dde_num_rounds=2,
        dde_num_reverse_rounds=2,
        subgraph_size=0,
    ):
        super().__init__()

        self.kg = kg_interface
        self.subgraph_size = subgraph_size
        self.text_encoder = text_encoder
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        emb_size = self.text_encoder.emb_size
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(dde_num_rounds, dde_num_reverse_rounds)

        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (dde_num_rounds + dde_num_reverse_rounds)

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        question,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        num_non_text_entities,
        q_id_tensor,
    ):
        unq_ent_id, id_tensor = torch.unique(
            torch.concat([h_id_tensor, t_id_tensor, q_id_tensor]), return_inverse=True
        )
        h_id_tensor_loc, t_id_tensor_loc, q_id_tensor_loc = torch.split(
            id_tensor.to(self.text_encoder.device),
            [len(h_id_tensor), len(t_id_tensor), len(q_id_tensor)],
        )
        text_entity_list = [self.kg.get_node_label(idx) for idx in unq_ent_id]
        relation_list, r_id_tensor_loc = torch.unique(r_id_tensor, return_inverse=True)
        r_id_tensor_loc = r_id_tensor_loc.to(self.text_encoder.device)
        text_relation_list = [self.kg.get_relation_label(idx) for idx in relation_list]

        q_emb, entity_embs, relation_embs = self.text_encoder(
            question, text_entity_list, text_relation_list
        )

        topic_entity_mask = torch.zeros(len(text_entity_list) + num_non_text_entities)
        topic_entity_mask[q_id_tensor_loc] = 1.0
        topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2).to(
            entity_embs.device
        )

        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(entity_embs.device)
                ).expand(num_non_text_entities, -1),
            ],
            dim=0,
        )
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([h_id_tensor_loc, t_id_tensor_loc], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor_loc, h_id_tensor_loc], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor_loc]

        h_triple = torch.cat(
            [h_q.expand(len(h_r), -1), h_e[h_id_tensor_loc], h_r, h_e[t_id_tensor_loc]],
            dim=1,
        )

        pred_triple_logits = self.pred(h_triple)

        if self.subgraph_size > 0:
            edge_ids = torch.topk(
                pred_triple_logits.flatten(),
                k=min(self.subgraph_size, pred_triple_logits.shape[0]),
            ).indices.to(h_id_tensor.device)
            subgraph_edges_h = h_id_tensor[edge_ids].numpy().tolist()
            subgraph_edges_t = t_id_tensor[edge_ids].numpy().tolist()
            subgraph_edges_r = r_id_tensor[edge_ids].numpy().tolist()
            top_subgraph_edges = list(
                zip(subgraph_edges_h, subgraph_edges_r, subgraph_edges_t)
            )
            return pred_triple_logits, top_subgraph_edges
        else:
            return pred_triple_logits, []
