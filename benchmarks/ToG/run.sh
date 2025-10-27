# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Jiashuo Sun.
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


python ToG/main_wiki.py \
  --dataset gtsqa \
  --wikikg_dir ../../data/ogbl_wikikg2 \
  --max_length 2048 \
  --temperature_exploration 0.4 \
  --temperature_reasoning 0 \
  --width 5 \
  --depth 4 \
  --remove_unnecessary_rel False \
  --LLM_type gpt-4o-mini \
  --num_retain_entity 5 \
  --prune_tools llm \
  --opeani_api_keys ${OPENAI_KEY}
