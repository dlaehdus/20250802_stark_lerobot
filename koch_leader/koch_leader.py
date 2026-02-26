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
# 로깅 기능을 위한 라이브러리
import time
# 시간 측정을 위한 라이브러리
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
# LeRobot 프로젝트의 장치 연결 관련 예외 클래스들 import
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
# 모터 제어를 위한 기본 클래스들 import
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
# Dynamixel 모터 전용 클래스들 import (Koch는 Dynamixel XM430-W350 모터 사용)
from ..teleoperator import Teleoperator
# 상위 디렉토리의 Teleoperator 베이스 클래스 import
from .config_koch_leader import KochLeaderConfig
# 같은 디렉토리의 Koch Leader 설정 클래스 import


logger = logging.getLogger(__name__)
# 현재 모듈명으로 로거 객체 생성



class KochLeader(Teleoperator):
# Teleoperator를 상속받는 KochLeader 클래스 정의 시작
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochLeaderConfig
    # 이 클래스가 사용할 설정 클래스 지정
    name = "koch_leader"
    # 이 텔레오퍼레이터의 이름 (타입 식별자)
    def __init__(self, config: KochLeaderConfig):
    # 생성자 메서드 정의, KochLeaderConfig 타입의 config 매개변수 받음
        super().__init__(config)
        # 부모 클래스(Teleoperator)의 생성자 호출
        self.config = config
        # 설정 객체를 인스턴스 변수로 저장
        self.bus = DynamixelMotorsBus(
        # Dynamixel 모터 버스 객체 생성 시작
            port=self.config.port,
            # 시리얼 포트 설정 (예: /dev/ttyACM0)
            # "--teleop.port=/dev/ttyACM0" 이걸로 찾는듯함
            motors={
                "shoulder_pan": Motor(10, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(11, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(12, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(13, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(14, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(15, "xm430-w350", MotorNormMode.RANGE_0_100),
            },
            # 모터 딕셔너리 정의 시작
            calibration=self.calibration,
            # 보정 데이터를 버스에 전달 (부모 클래스에서 로드됨)
        )

    # 프로퍼티 데코레이터 (메서드를 속성처럼 사용 가능)
    @property
    def action_features(self) -> dict[str, type]:
    # 액션 특성 반환 메서드 (어떤 데이터를 출력할지 정의)
        return {f"{motor}.pos": float for motor in self.bus.motors}
        # 각 모터의 위치를 float 타입으로 반환 (예: "shoulder_pan.pos": float)

    @property
    def feedback_features(self) -> dict[str, type]:
    # 피드백 특성 정의 (Leader는 피드백 없음)
        return {}
        # 빈 딕셔너리 반환 (피드백 기능 없음)

    @property
    def is_connected(self) -> bool:
    # 연결 상태 확인 프로퍼티
        return self.bus.is_connected
        # 버스의 연결 상태를 반환

    def connect(self, calibrate: bool = True) -> None:
    # 연결 메서드, 기본적으로 보정도 함께 수행
        if self.is_connected:
        # 이미 연결되어 있으면
            raise DeviceAlreadyConnectedError(f"{self} already connected")
            # "이미 연결됨" 예외 발생
        self.bus.connect()
        # 실제 하드웨어 연결 수행
        if not self.is_calibrated and calibrate:
        # 보정되지 않았고 보정 옵션이 True면
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            # 보정 파일 불일치 또는 없음을 로그에 기록
            self.calibrate()
            # 보정 과정 시작
        self.configure()
        # 모터 설정 적용
        logger.info(f"{self} connected.")
        # 연결 완료를 로그에 기록


    @property
    def is_calibrated(self) -> bool:
    # 보정 상태 확인 프로퍼티
        return self.bus.is_calibrated
        # 버스의 보정 상태 반환


    # 이부분이 제일 중요한 핵심부분
    def calibrate(self) -> None:
    # 보정 메서드 정의
        if self.calibration:
        # 기존 보정 파일이 있으면

            # 이러면 엔터 치라고 안뜸
            # Calibration file exists, ask user whether to use it or run new calibration
            # user_input = input(
            #     f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            # )
            # 자동으로 엔터 누른걸로 처리
            user_input = ""
            # 사용자에게 기존 보정 사용 또는 새 보정 선택 요청
            if user_input.strip().lower() != "c":
            # 'c'가 아니면 (기존 보정 사용)
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                # 기존 보정 파일 적용 로그
                self.bus.write_calibration(self.calibration)
                # 기존 보정 데이터를 모터에 적용
                return
                # 메서드 종료
        logger.info(f"\nRunning calibration of {self}")
        # 새 보정 시작 로그
        self.bus.disable_torque()
        # 모든 모터의 토크 비활성화 (자유롭게 움직일 수 있도록)
        for motor in self.bus.motors:
            if motor == "gripper":
                # 그리퍼는 전류 기반 위치 모드
                self.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
            else:
                # 다른 모터들은 확장 위치 모드
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        # 사용자에게 로봇을 휴식 위치로 움직이라고 지시
        homing_offsets = self.bus.set_half_turn_homings()
        # 현재 위치를 기준점으로 설정하여 호밍 오프셋 계산
        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        # 완전 회전 가능한 모터들 정의
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        # 범위를 측정해야 하는 모터들 (완전 회전 불가능한 모터들)
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        # 완전 회전 모터 외의 모터들을 전체 범위로 움직이라고 지시
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        # 각 모터의 최소/최대 위치 기록
        for motor in full_turn_motors:
        # 완전 회전 가능한 모터들에 대해
            range_mins[motor] = 0
            range_maxes[motor] = 524288
            # range_mins[motor] = 0
            # range_maxes[motor] = 4095
            # 전체 범위(0~8095)로 설정 (확장 모드에서 2회전 가능)
        self.calibration = {}
        # 보정 데이터 딕셔너리 초기화
        for motor, m in self.bus.motors.items():
        # 각 모터와 모터 객체에 대해
            self.calibration[motor] = MotorCalibration(
            # 모터별 보정 데이터 생성
                id=m.id,
                # 모터 아이디
                # drive_mode=drive_modes[motor],
                drive_mode=0,
                # 드라이브 방향 정방향 모드
                homing_offset=homing_offsets[motor],
                # 호밍 오프셋 (중간 위치 기준점)
                range_min=range_mins[motor],
                # 최소 위치값
                range_max=range_maxes[motor],
                # 최대 위치값
            )
            # MotorCalibration 객체 생성 완료

        self.bus.write_calibration(self.calibration)
        # 보정 데이터를 모터들에 적용
        self._save_calibration()
        # 보정 데이터를 파일로 저장
        logger.info(f"Calibration saved to {self.calibration_fpath}")
        # 보정 파일 저장 완료 로그
        
    def configure(self) -> None:
    # 모터 설정 메서드
        self.bus.disable_torque()
        # 설정 변경을 위해 토크 비활성화
        self.bus.configure_motors()
        # 기본 모터 설정 적용
        for motor in self.bus.motors:
        # 모든 모터에 대해
            if motor != "gripper":
                # 그리퍼가 아닌 모터들은
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
                # 확장 위치 모드로 설정 (360도 이상 회전 가능)
        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        # 그리퍼는 전류 기반 위치 제어 모드로 설정 (힘 제한)
        # self.bus.enable_torque("gripper")
        if self.is_calibrated:
        # 보정이 완료되었으면
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)
            # 그리퍼를 열린 위치로 설정


    def setup_motors(self) -> None:
    # 모터 ID 설정 메서드 (setup_motors.py에서 호출됨)
        for motor in reversed(self.bus.motors):
        # 모터들을 역순으로 (gripper → shoulder_pan 순서)
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            # 특정 모터만 연결하고 Enter를 누르라고 지시
            self.bus.setup_motor(motor)
            # 해당 모터의 ID와 baudrate 설정
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")
            # 설정 완료 메시지 출력

    def get_action(self) -> dict[str, float]:
    # 현재 로봇 상태 읽기 메서드
        if not self.is_connected:
        # 연결되지 않았으면
            raise DeviceNotConnectedError(f"{self} is not connected.")
            # 연결 안됨 예외 발생
        start = time.perf_counter()
        # 성능 측정 시작 시간 기록
        action = self.bus.sync_read("Present_Position")
        # 모든 모터의 현재 위치를 동시에 읽기
        action = {f"{motor}.pos": val for motor, val in action.items()}
        # 딕셔너리 형태로 변환 (예: {"shoulder_pan.pos": 1024})
        dt_ms = (time.perf_counter() - start) * 1e3
        # 읽기 시간을 밀리초로 계산
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        # 읽기 성능을 디버그 로그로 기록
        return action
        # 위치 데이터 반환

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
