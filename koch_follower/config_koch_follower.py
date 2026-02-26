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

from dataclasses import dataclass, field
# dataclass와 field: Python의 데이터 클래스 기능을 사용해 간단한 설정 객체를 만듦.

from lerobot.cameras import CameraConfig
# CameraConfig: 카메라 설정 클래스 (lerobot.cameras 모듈에서).

from ..config import RobotConfig
# LeRobot의 기본 로봇 config 클래스 (상속받음).

@RobotConfig.register_subclass("koch_follower")
# @RobotConfig.register_subclass("koch_follower"): 이 config를 "koch_follower"라는 이름으로 LeRobot에 등록.
# 크립트에서 --robot-type koch_follower처럼 불러올 수 있음.
@dataclass
# dataclass: 필드를 자동으로 초기화하고, equality 등을 제공.
class KochFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    # port: str: Dynamixel 모터 버스 연결 포트 (e.g., "/dev/ttyUSB0"). 필수 매개변수로, 초기화 시 반드시 제공해야 함.
    disable_torque_on_disconnect: bool = True
    # disable_torque_on_disconnect: bool = True: 연결 해제 시 모터 토크를 비활성화할지 여부. 안전을 위해 기본 True (모터가 자유롭게 움직이게 함).
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None
    # max_relative_target: int | None = None: 상대적 목표 위치의 최대 크기 제한 (안전용). 
    # 양수 스칼라(모든 모터 동일) 또는 리스트(모터별)로 설정. None이면 제한 없음. 
    # 과도한 움직임을 방지해 하드웨어 손상을 막음.
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # cameras: dict[str, CameraConfig] = field(default_factory=dict): 카메라 설정 딕셔너리 (키: 카메라 이름, 값: CameraConfig 객체).
    # 기본 빈 딕셔너리. field()로 지연 초기화.
    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
    # use_degrees: bool = False: 이전 정책/데이터셋과의 호환성을 위해 각도(degrees) 단위를 사용할지 여부.
    # 기본 False (정규화된 범위 사용, e.g., -100~100).
