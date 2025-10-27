The code in this repository implements SubgraphRAG (https://arxiv.org/abs/2410.20724).

See `train/` for training script. It requires first computing the `scores.pkl` files for train and test set with [compute_neighs_and_sp.py](../../synth_kgqa/compute_neighs_and_sp.py).

```
python3 train/train_subgraphrag.py --wikikg_dir <path to KG directory>  --train_data_path <path to scores.pkl output for train set> --valid_data_path <path to scores.pkl output for test set>
```
