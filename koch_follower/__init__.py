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

from .config_koch_follower import KochFollowerConfig
from .koch_follower import KochFollower
# 이 코드는 Koch follower 로봇을 LeRobot에 통합해, 텔레오퍼레이션(teleoperation), 데이터 수집, 정책 학습을 지원합니다.
# Dynamixel 기반으로 저비용 로봇 팔에 최적화.