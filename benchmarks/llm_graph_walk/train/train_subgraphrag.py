# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Graph-COM

# code adapted from https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/train.py

import logging
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from llm_graph_walk import graph
from llm_graph_walk.sample_subgraphrag import SampleSubgraphRAG
from llm_graph_walk.text_encoder import TextEncoderSubgraphRAG
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from accelerate import Accelerator

logger = logging.getLogger(__name__)


def create_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_step(logger, log: dict, step: int, wandb_log: bool = False):
    logger.info(f"Step {step}")
    for k, v in log.items():
        logger.info(f"   {k}: {v:.5f}")
    if wandb_log:
        wandb.log(log, step=step)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RetrieverDataset(Dataset):
    def __init__(self, data_path, question_style=None, max_size=None, use_sp=False):
        with open(data_path, "rb") as f:
            preprocessed_data = pickle.load(f)
        self.preprocessed_data = preprocessed_data
        if question_style == "rand":
            for i in range(len(self.preprocessed_data)):
                if np.random.uniform() < 0.5:
                    self.preprocessed_data[i]["question"] = self.preprocessed_data[i][
                        "paraphrased_question"
                    ]

        if max_size:
            self.preprocessed_data = [
                self.preprocessed_data[i]
                for i in np.random.randint(len(preprocessed_data), size=(max_size,))
            ]
        self.use_sp = use_sp

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        sample = self.preprocessed_data[idx]
        return (
            sample["question"],
            torch.tensor(sample["h_id_list"]),
            torch.tensor(sample["r_id_list"]),
            torch.tensor(sample["t_id_list"]),
            0,  # backward compatibility
            torch.tensor(sample["topic_entity_id"]),
            torch.tensor(
                sample["triple_scores_sp" if self.use_sp else "triple_scores"]
            ),
            torch.tensor(sample["answer_id"]),
        )


@torch.no_grad()
def run_validation(data_loader, model, valid_k_list):
    metric_dict = defaultdict(list)
    eval_start = time.time()
    for sample in tqdm(data_loader):
        pred_triple_logits, _ = model(*sample[:-2])
        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits.flatten(), descending=True
        )
        target_triple_ids = sample[-2].to(sorted_triple_ids_pred.device).nonzero()
        a_entity_id_list = sample[-1].unsqueeze(-1)
        ranks = torch.where(
            sorted_triple_ids_pred == target_triple_ids,
            torch.arange(
                1, sorted_triple_ids_pred.shape[0] + 1, device=target_triple_ids.device
            ),
            torch.inf,
        )  # (n_target_triples, subgraph_size)
        num_correct_triples = ranks.shape[0]
        rank = ranks.min(dim=-1)[0].cpu()
        metric_dict["gt_triple_mrr"].append(
            torch.mean(
                1.0
                / torch.maximum(
                    torch.tensor(1.0),
                    rank - num_correct_triples + 1,
                )
            ).item()
        )

        for k in valid_k_list:
            metric_dict[f"gt_triple_recall@{k}"].append(
                (rank < k).sum().item() / num_correct_triples
            )
            retrieved_triples = sorted_triple_ids_pred[:k].cpu()
            recall_answ = torch.any(
                torch.concat(
                    [sample[1][retrieved_triples], sample[3][retrieved_triples]]
                )
                == a_entity_id_list,
                dim=-1,
            )
            metric_dict[f"answer_recall@{k}"].append(
                recall_answ.numpy().sum().item() / len(a_entity_id_list)
            )

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val).item()

    metric_dict["eval_duration"] = time.time() - eval_start
    return metric_dict


def main(args):
    device = args.device
    # accelerator = Accelerator()
    set_seed(args.seed)
    create_logger()
    if args.wandb:
        wandb.init(
            project=f"SubgraphRAG_training",
            config=args,
        )

    logger.info("Loading data")
    train_set = RetrieverDataset(
        args.train_data_path, question_style="rand", use_sp=args.use_sp
    )
    val_set = RetrieverDataset(args.valid_data_path)
    train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False)

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
        edge_ids, relation_types, node_labels, relation_labels, with_neigh_dict=False
    )

    logger.info("Instantiating model")
    text_encoder = TextEncoderSubgraphRAG(args.text_encoder_path, device=device)
    model = SampleSubgraphRAG(
        graph.KGInterfaceFromGraph(knowledge_graph),
        text_encoder,
        args.topic_pe,
        args.dde_num_rounds,
        args.dde_num_reverse_rounds,
        0,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    logger.info(
        f"Model created -- {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters",
    )

    save_dir = os.path.join("checkpoints", args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    # model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    step = -1
    partial_loss = []
    model.train()
    for epoch in range(1, args.train_epochs + 1):
        logger.info(f"Training -- epoch {epoch}")
        for sample in tqdm(train_loader):
            step += 1
            batch_start = time.time()
            pred_triple_logits, _ = model(*sample[:-2])
            target_triple_probs = sample[-2].to(device).unsqueeze(-1)
            # target_triple_probs = sample[-2].reshape(-1,1).to(device)
            loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, target_triple_probs
            )
            optimizer.zero_grad()
            loss.backward()
            # accelerator.backward(loss)
            optimizer.step()
            partial_loss.append(loss.item())
            step_duration = time.time() - batch_start

            if step % args.log_interval == 0 or step % args.valid_interval == 0:
                log_dict = {
                    "loss": np.mean(partial_loss),
                    "subgraph_size": sample[1].numel(),
                    "step_duration": step_duration,
                }
                partial_loss = []

                if step % args.valid_interval == 0:
                    model.eval()
                    logger.info("Running validation")
                    log_dict.update(run_validation(val_loader, model, args.valid_k))
                    model.train()

                log_step(logger, log_dict, step, args.wandb)

        if args.save_path and (epoch % 10 == 0 or epoch == args.train_epochs):
            state_dict = {"config": args, "model_state_dict": model.state_dict()}
            torch.save(
                state_dict,
                os.path.join(save_dir, f"checkpoint_ep_{epoch}.pth"),
            )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["webqsp", "synthetic", "synthetic_def"],
        default="synthetic_def",
        # default="webqsp",
        help="Dataset name",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="preprocessed train dataset",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        help="preprocessed validation dataset",
    )
    parser.add_argument(
        "--kg",
        type=str,
        choices=["wikikg2", "wikidata"],
        default="wikikg2",
        help="KG name",
    )
    parser.add_argument(
        "--wikikg_dir",
        type=str,
        help="directory containing the processed wikikg2",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Alibaba-NLP/gte-large-en-v1.5",
        help="Text encoder HF path",
    )
    parser.add_argument("--topic_pe", type=bool, default=True)
    parser.add_argument("--dde_num_rounds", type=int, default=2)
    parser.add_argument("--dde_num_reverse_rounds", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=70)
    parser.add_argument("--valid_interval", type=int, default=6000)
    parser.add_argument("--valid_k", type=int, nargs="*", default=[10, 200])
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_sp", action="store_true")
    args = parser.parse_args()

    main(args)
