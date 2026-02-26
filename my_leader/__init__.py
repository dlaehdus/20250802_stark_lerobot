#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config_my_leader import myLeaderConfig
from .my_leader import myLeader

# .(점): 현재 패키지 내에서 모듈을 찾겠다는 의미
# config_my_leader: 같은 디렉토리에 있는 config_my_leader.py 파일
# my_leader: 같은 디렉토리에 있는 my_leader.py 파일