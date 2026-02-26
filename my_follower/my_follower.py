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

import logging
# logging, time: 로그와 타이밍 측정.
import time
from functools import cached_property
# cached_property: 속성을 캐싱해 반복 계산 피함.
from typing import Any
# typing.Any: 타입 힌트.
# LeRobot 모듈: 카메라 유틸, 에러 클래스, 모터 클래스, Dynamixel 버스, 운영 모드.
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from ..robot import Robot
# Robot: 기본 로봇 클래스 (상속).
from ..utils import ensure_safe_goal_position
# ensure_safe_goal_position: 안전한 목표 위치 보장 유틸 (상대적 움직임 제한).
from .config_my_follower import myFollowerConfig
# KochFollowerConfig: 같은 디렉토리의 config 임포트.

logger = logging.getLogger(__name__)
# logger: 이 모듈의 로거 설정.

class myFollower(Robot):
# 클래스 정의: Robot 상속. docstring은 Koch 로봇 버전 설명 (v1.0과 v1.1 링크).
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = myFollowerConfig
    # config_class: 연결된 config 클래스.
    name = "my_follower"
    # name: 로봇 식별자 ("koch_follower").

    def __init__(self, config: myFollowerConfig):
        super().__init__(config)
        # super()로 부모 초기화.
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        # norm_mode_body: config에 따라 각도(DEGREES) 또는 정규화 범위(RANGE_M100_100) 선택.
        self.bus = DynamixelMotorsBus(
        # self.bus: DynamixelMotorsBus 인스턴스. 6개 모터 정의 (ID 0~5, 모델 "xm430-w350"). 그리퍼는 0~100 범위. 캘리브레이션 로드.
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(0, "xm430-w350", norm_mode_body),
                "shoulder_lift": Motor(1, "xm430-w350", norm_mode_body),
                "elbow_flex": Motor(2, "xm430-w350", norm_mode_body),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # self.cameras: config.cameras에서 카메라 인스턴스 생성.

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}
    # _motors_ft: 모터 위치 피처 (e.g., "shoulder_pan.pos": float).

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    # _cameras_ft: 카메라 이미지 shape (높이, 너비, 채널 3)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}
    # observation_features: 관찰 피처 (모터 + 카메라). cached_property로 캐싱.

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
    # action_features: 액션 피처 (모터 위치만).

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
    # is_connected: 버스와 모든 카메라 연결 상태 확인.

    def connect(self, calibrate: bool = True) -> None:
    # connect: 연결 메서드.
    # 이미 연결 시 에러 발생.
    # 버스 연결.
    # 캘리브레이션 필요 시 calibrate() 호출.
    # 카메라 연결.
    # configure() 호출 (모터 설정).
    # 로그 출력.
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated
    # is_calibrated: 버스의 캘리브레이션 상태 반환.

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()

        # 그리퍼를 제외한 모든 모터를 확장 위치 모드로 변환함
        # EXTENDED_POSITION: 360도 이상 회전 가능 (일반 관절용)
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "elbow_flex"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )

        # 이 값을 수정하면 됌
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")
    # calibrate: 캘리브레이션 프로세스.
    # 기존 파일 있으면 사용자 입력으로 선택 (ENTER: 사용, 'c': 새로).
    # 토크 비활성화, 운영 모드 설정 (EXTENDED_POSITION: 360도 이상 회전 가능).
    # 사용자 입력으로 중간 위치 이동, homing offset 설정.
    # 전체 턴 모터(shoulder_pan, wrist_roll)는 범위 0~4095 고정.
    # 다른 모터는 움직임 범위 기록.
    # MotorCalibration 객체 생성 및 저장 (파일과 모터에 쓰기).

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
            # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
            # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point
            for motor in self.bus.motors:
                # if motor != "gripper":
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Use 'position control current based' for gripper to be limited by the limit of the current. For
            # the follower gripper, it means it can grasp an object without forcing too much even tho, its
            # goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with
            # our finger to make it move, and it will move back to its original target position when we
            # release the force.

            # 그리퍼만 설저함
            # CURRENT_POSITION: 전류 기반 위치 제어 (그리퍼용, 과도한 힘 방지)
            # self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor

            # PID제어부분
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)
    # configure: 모터 설정.
    # 토크 비활성화 컨텍스트.
    # 모터 기본 설정.
    # 그리퍼 제외 모든 모터: EXTENDED_POSITION 모드 (360도 초과 회전 가능, 조립 오류 방지).
    # 그리퍼: CURRENT_POSITION 모드 (전류 기반 위치 제어, 과도한 힘 방지, 트리거처럼 사용 가능).
    # PID 값 설정 (elbow_flex 모터: P=1500, I=0, D=600). 상태-액션 갭 최소화. TODO: 자동화 필요.

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
    # setup_motors: 모터 ID 설정 프로세스. 각 모터를 순차적으로 연결해 ID 할당 (역순으로).

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    # get_observation: 관찰 데이터 수집.
    # 연결 확인.
    # 모터 위치 동기 읽기 (Present_Position), 딕셔너리로 변환 (e.g., "shoulder_pan.pos": 값).
    # 타이밍 로그 (디버그).
    # 카메라 이미지 비동기 읽기, 딕셔너리에 추가.
    # 반환: 모터 위치 + 카메라 이미지 딕셔너리.

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    # send_action: 액션 전송 (목표 위치 설정).
    # 연결 확인.
    # 액션에서 ".pos" 접미사 제거해 goal_pos 딕셔너리 생성.
    # max_relative_target 있으면 현재 위치 읽고, 안전 클리핑 (ensure_safe_goal_position 사용). FPS 저하 주의.
    # 동기 쓰기 (Goal_Position).
    # 반환: 실제 전송된 액션 (클리핑된 경우 다를 수 있음).

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
    # disconnect: 연결 해제.
    # 버스 해제 (토크 비활성화 옵션).
    # 카메라 해제.
    # 로그 출력.
