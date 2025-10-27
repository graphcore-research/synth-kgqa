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

# adapted from https://github.com/GasolSun36/ToG/blob/main/eval/utils.py


def exact_match(response, answers):
    clean_result = response.strip().replace(" ", "").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ", "").lower()
        if (
            clean_result == clean_answer
            or clean_result in clean_answer
            or clean_answer in clean_result
        ):
            return True
    return False
