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
Helper to recalibrate your device (robot or teleoperator).

Example:

```shell
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```
"""

import logging
# 로깅 라이브러리 import
from dataclasses import asdict, dataclass
# 데이터클래스 관련 함수들 import
from pprint import pformat
#  예쁜 출력 포맷팅 함수 import

import draccus
# 명령줄 인수 파싱 라이브러리 import

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
# OpenCV 카메라 설정 클래스 import (사용하지 않아도 import)
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
# RealSense 카메라 설정 클래스 import

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    my_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
# 로봇 관련 클래스들과 팩토리 함수 import

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    homunculus,
    koch_leader,
    my_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
# 텔레오퍼레이터 관련 클래스들과 팩토리 함수 import


from lerobot.utils.utils import init_logging
# 로깅 초기화 함수 import


# 데이터클래스 데코레이터
@dataclass
class CalibrateConfig:
# 보정 설정을 담을 클래스 정의
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


# 명령줄 인수를 자동으로 파싱하는 데코레이터
@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
# 보정 메인 함수 정의, CalibrateConfig 타입의 cfg 매개변수
    init_logging()
    # 로깅 시스템 초기화
    logging.info(pformat(asdict(cfg)))
    # 설정 내용을 예쁘게 포맷해서 로그로 출력

    # 해당 로봇이나 마스터나 슬레이브를 찾음
    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)

    device.connect(calibrate=False)
    # 장치에 연결 (보정은 나중에 별도로 실행하므로 calibrate=False)
    device.calibrate()
    # 보정 과정 실행 (실제 보정 작업 수행)
    device.disconnect()
    # 장치 연결 해제

def main():
    calibrate()


if __name__ == "__main__":
    main()
