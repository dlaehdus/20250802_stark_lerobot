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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
# 예쁜 출력 포맷팅 함수 import

import draccus
# 명령줄 인수 파싱 라이브러리 import
import rerun as rr
# Rerun 3D 시각화 라이브러리 import

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
# 카메라 설정 클래스들 import


from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    my_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
# 로봇 관련 클래스들과 팩토리 함수 import


from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    my_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
# 텔레오퍼레이터 관련 클래스들과 팩토리 함수 import


from lerobot.utils.robot_utils import busy_wait
# 정밀한 타이밍 제어를 위한 busy_wait 함수
from lerobot.utils.utils import init_logging, move_cursor_up
# 로깅 초기화와 터미널 커서 제어 함수들
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
# Rerun 시각화 초기화와 데이터 로깅 함수들



@dataclass
class TeleoperateConfig:
# 텔레오퍼레이션 설정을 담을 데이터클래스 정의
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    # 제어 주파수 설정 (기본: 60Hz)
    teleop_time_s: float | None = None
    # 텔레오퍼레이션 지속시간 (기본: 무제한)
    # Display all cameras on screen
    display_data: bool = False
    # 실시간 데이터 표시 여부 (Rerun 뷰어 사용)


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
# 텔레오퍼레이션 메인 루프 함수 정의
    display_len = max(len(key) for key in robot.action_features)
    # 출력 포맷팅을 위한 최대 키 길이 계산
    start = time.perf_counter()
    # 시작 시간 기록 (지속시간 제한용)
    while True:
    # 무한 루프 시작
        loop_start = time.perf_counter()
        # 루프 시작 시간 기록 (성능 측정용)
        action = teleop.get_action()
        # 텔레오퍼레이터에서 현재 동작 읽기 (조작자 로봇의 관절 위치들)
        if display_data:
        # 시각화가 활성화되어 있으면
            observation = robot.get_observation()
            # 로봇에서 현재 상태 읽기 (모터 위치 + 카메라 이미지들)
            log_rerun_data(observation, action)
            # Rerun 뷰어에 데이터 전송 (3D 시각화용)

        robot.send_action(action)
        # 로봇에 동작 명령 전송 (텔레오퍼레이터 동작을 팔로워가 따라함)
        dt_s = time.perf_counter() - loop_start
        # 지금까지 걸린 시간 계산
        busy_wait(1 / fps - dt_s)
        # 정확한 주파수 유지를 위한 대기 (예: 60Hz = 16.67ms 주기)
        loop_s = time.perf_counter() - loop_start
        # 전체 루프 시간 계산
        print("\n" + "-" * (display_len + 10))
        # 구분선 출력
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        # 테이블 헤더 출력 (NAME, NORM 컬럼)
        for motor, value in action.items():
        # 각 모터의 동작값에 대해
            print(f"{motor:<{display_len}} | {value:>7.2f}")
            # 모터 이름과 정규화된 위치값 출력
            # 예: shoulder_pan | -0.45
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        # 루프 시간과 주파수 출력
        if duration is not None and time.perf_counter() - start >= duration:
        # 지속시간이 설정되어 있고 시간이 지났으면
            return
            # 함수 종료

        move_cursor_up(len(action) + 5)
        # 터미널 커서를 위로 이동해서 같은 자리에 업데이트된 정보 출력


# 명령줄 인수 자동 파싱 데코레이터
@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
# 메인 텔레오퍼레이션 함수
    init_logging()
    # 로깅 시스템 초기화
    logging.info(pformat(asdict(cfg)))
    # 설정 정보를 로그로 출력
    if cfg.display_data:
    # 시각화가 활성화되어 있으면
        _init_rerun(session_name="teleoperation")
        # Rerun 3D 뷰어 초기화
    teleop = make_teleoperator_from_config(cfg.teleop)
    # 텔레오퍼레이터 객체 생성 (조작자 로봇)
    robot = make_robot_from_config(cfg.robot)
    # 로봇 객체 생성 (팔로워 로봇)

    teleop.connect()
    # 텔레오퍼레이터에 연결 (조작자 로봇 하드웨어 연결)
    robot.connect()
    # 로봇에 연결 (팔로워 로봇 하드웨어 + 카메라 연결)

    try:
    # 예외 처리 시작
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
    # 사용자가 Ctrl+C로 중단하면
        pass
        # 정상 종료 처리
    finally:
        if cfg.display_data:
        # 시각화가 활성화되어 있었으면
            rr.rerun_shutdown()
            # Rerun 뷰어 종료
        teleop.disconnect()
        robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
