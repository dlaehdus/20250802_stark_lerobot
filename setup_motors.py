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
Helper to set motor ids and baudrate.

Example:

```shell
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751
```
"""

from dataclasses import dataclass
# Python의 데이터클래스 데코레이터를 가져오기(클래스를 쉽게 만들기 도구)
import draccus
# draccus 클래스 import (훈련줄 인수를 자동으로 파싱하는 도구)

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    my_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
# lerobot의 로봇 모듈에서 여러 클래스들을 import
# 내가 새로 로봇을 만들면 이곳에 추가해야할듯 함
# 이건 slave즉 따라가는 놈의 설정

from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    koch_leader,
    my_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
# lerobot의 teleoperators 모듈에서 다양한 클래스들을 import
# 내가 새로 로봇을 만들면 이곳에 추가해야할듯 함
# 이건 master즉 조종기의 설정

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "my_follower",
    "my_leader",
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "lekiwi",
]
# 호환 가능한 장치들의 목록 시작



# 마스터와 슬레이브 둘중에 하나만 있을때 실행돼는 설정 이외에는 오류발생
@dataclass
# 데이터클래스 데코레이터: 아래 클래스를 자동으로 설정 클래스로 만들어줌
class SetupConfig:
# 설정을 담을 클래스 정의 시작
    teleop: TeleoperatorConfig | None = None
    # 텔레오퍼레이터 설정: TeleoperatorConfig 유형이 없거나 None, 그런 것은 None입니다.
    # 조종기 설정
    robot: RobotConfig | None = None
    # 로터리 설정: RobotConfig 유형에 따라 None, 그런 것은 None입니다.
    # 따라가는놈 노예 설정
    def __post_init__(self):
    # 클래스 생성 후 자동으로 호출되는 메서드 정의
        if bool(self.teleop) == bool(self.robot):
        # 텔레오프와 로봇 둘 다 없으면 다 제외 (XOR 논리)
            raise ValueError("Choose either a teleop or a robot.")
            # 잘못된 발생: "레오텔퍼레이터나 치료 중 선택하세요"
        self.device = self.robot if self.robot else self.teleop
        # 로봇이 있으면 로봇을, 없으면 텔레옵을 장치에 저장



# draccus 데코레이터: 함수열 인수를 자동으로 파싱해서 함수에 전달
@draccus.wrap()
def setup_motors(cfg: SetupConfig):
# 메인 정의: SetupConfig 유형의 cfg별로 별도를 받아들입니다.
    if cfg.device.type not in COMPATIBLE_DEVICES:
    # 선택 장치 타입이 지원 장치 목록에 없으면
        raise NotImplementedError
        # 불법행위: "구현되지 않은 장치입니다."
    if isinstance(cfg.device, RobotConfig):
    # 선택한 장치가 RobotConfig 타입이면 (즉, 로봇이면)
        device = make_robot_from_config(cfg.device)
        # 로터리 설정을 생성합니다.
        # make_robot_from_config 이걸로 어떤 로봇인지 판단하고 해당 로봇의 설정으로 들어가서
        # 해당 아이디들을 설정하는것으로 보임
    else:
    # 그렇지 않으면 (즉, 텔레오퍼레이터면)
        device = make_teleoperator_from_config(cfg.device)
        # 텔레오퍼 설정으로부터 텔레오퍼 생성 생성
    device.setup_motors()
    # 생성된 장치를 설정하는 setup_motors 메서드 호출(실제 모터 설정 작업 수행)


def main():
    setup_motors()


if __name__ == "__main__":
    main()
