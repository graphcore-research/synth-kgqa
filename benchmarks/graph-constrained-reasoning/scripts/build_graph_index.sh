# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Linhao Luo (Raymond)


# For training
SPLIT=train_def_scores.pkl # output of compute_neighs_and_sp.py
N_PROCESS=8
python workflow/build_shortest_path_index.py --n ${N_PROCESS} --output_path data/shortest_path_index --kg-path ../../data/ogbl_wikikg2 --data_path ../../GTSQA --split ${SPLIT}

python workflow/build_shortest_path_index.py --n ${N_PROCESS} --output_path data/shortest_path_index_sp --kg-path ../../data/ogbl_wikikg2 --data_path ../../GTSQA --split ${SPLIT} --use_sp

# For evaluation

# DATA_PATH="RoG-webqsp RoG-cwq"
# SPLIT=test
# N_PROCESS=8
# HOP=2 # 3
# for DATA_PATH in ${DATA_PATH}; do
#     python workflow/build_graph_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --K ${HOP}
# done