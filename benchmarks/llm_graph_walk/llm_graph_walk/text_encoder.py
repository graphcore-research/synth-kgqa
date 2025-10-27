# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Graph-COM

# code adapted from https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/model/text_encoders/gte_large_en.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    def __init__(
        self, model_path: str, device: str, model_config={}, tokenizer_config={}
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
        self.model = AutoModel.from_pretrained(model_path, **model_config).to(device)
        self.model.init_weights()
        self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def emb_size(self):
        return self.embed("test").shape[-1]

    def embed(self, text_list):
        raise NotImplementedError


class TextEncoderSubgraphRAG(TextEncoder):
    def __init__(
        self,
        model_path="Alibaba-NLP/gte-large-en-v1.5",
        device="cuda:0",
        normalize=True,
    ):
        model_config = {
            "trust_remote_code": True,
            "unpad_inputs": True,
            "use_memory_efficient_attention": True,
        }
        super().__init__(model_path, device, model_config)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        batch_dict = self.tokenizer(
            text_list,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)

        return q_emb, entity_embs, relation_embs
