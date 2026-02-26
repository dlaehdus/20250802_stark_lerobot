#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
# 데이터클래스 데코레이터 임포트
# __init__, __repr__, __eq__ 등의 메서드를 생성해주는 기능
from ..config import TeleoperatorConfig
# 상위 디렉토리의 config 모듈에서 TeleoperatorConfig 기본 클래스 임포트

@TeleoperatorConfig.register_subclass("my_leader")
# 이 클래스를 "my_leader"라는 이름으로 LeRobot 시스템에 등록
# 명령줄에서 --teleop.type=my_leader로 사용 가능하게 만듦

# 데이터클래스 데코레이터 적용
@dataclass
class myLeaderConfig(TeleoperatorConfig):
# TeleoperatorConfig를 상속받아 텔레오퍼레이터 공통 기능 포함
    # Port to connect to the arm
    port: str
    # 로봇 팔에 연결할 포트 경로 (예: "/dev/ttyACM0")
    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    # gripper_open_pos: float = 50.0
    # 토크 모드에서 그리퍼 모터의 기본 위치값