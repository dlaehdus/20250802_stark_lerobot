# !/usr/bin/env python
# 
# 카메라 검색: OpenCV와 RealSense 카메라를 자동으로 찾기
# 
# 정보 출력: 각 카메라의 상세 정보 표시 (해상도, FPS 등)
# 
# 이미지 캡처: 모든 카메라에서 동시에 이미지 촬영
# 
# 파일 저장: outputs/captured_images/ 폴더에 PNG 파일로 저장
# 
# 멀티스레딩: 여러 카메라를 동시에 처리하여 성능 향상
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
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot.find_cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
# 명령줄 인수 파싱용 라이브러리
import concurrent.futures
# 멀티스레딩용 라이브러리 (여러 카메라를 동시에 처리)
import logging
# 로깅 시스템
import time
# 시간 측정용
from pathlib import Path
# 파일 경로 처리용
from typing import Any
# 타입 힌트용
import numpy as np
# 수치 연산 라이브러리 (이미지 데이터 처리)
from PIL import Image
# 이미지 저장용 라이브러리
from lerobot.cameras.configs import ColorMode
# 카메라 색상 모드 설정
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
# OpenCV 카메라 관련 클래스들
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
# Intel RealSense 카메라 관련 클래스들


logger = logging.getLogger(__name__)
# 현재 모듈용 로거 생성



def find_all_opencv_cameras() -> list[dict[str, Any]]:
# 모든 OpenCV 호환 카메라를 찾는 함수 정의
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    # OpenCV 카메라 정보를 저장할 빈 리스트 생성
    logger.info("Searching for OpenCV cameras...")
    # OpenCV 카메라 검색 시작 로그
    try:
    # 예외 처리 시작
        opencv_cameras = OpenCVCamera.find_cameras()
        # OpenCVCamera 클래스의 find_cameras() 메서드로 카메라 검색
        for cam_info in opencv_cameras:
        # 찾은 각 카메라에 대해
            all_opencv_cameras_info.append(cam_info)
            # 카메라 정보를 리스트에 추가
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
        # 찾은 OpenCV 카메라 개수 로그 출력
    except Exception as e:
    # 예외 발생 시
        logger.error(f"Error finding OpenCV cameras: {e}")
        # 오류 메시지 로그 출력
    return all_opencv_cameras_info
    # OpenCV 카메라 정보 리스트 반환


def find_all_realsense_cameras() -> list[dict[str, Any]]:
# 모든 RealSense 카메라를 찾는 함수 정의
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    # RealSense 카메라 정보를 저장할 빈 리스트 생성
    logger.info("Searching for RealSense cameras...")
    # RealSense 카메라 검색 시작 로그
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        # RealSenseCamera 클래스로 카메라 검색
        for cam_info in realsense_cameras:
        # 찾은 각 RealSense 카메라에 대해
            all_realsense_cameras_info.append(cam_info)
            # 카메라 정보를 리스트에 추가
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
        # 찾은 RealSense 카메라 개수 로그 출력
    except ImportError:
    # pyrealsense2 라이브러리가 없는 경우
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
        # RealSense 라이브러리 없음 경고 메시지
    except Exception as e:
    # 기타 예외 발생 시
        logger.error(f"Error finding RealSense cameras: {e}")
        # 오류 메시지 로그 출력

    return all_realsense_cameras_info
    # RealSense 카메라 정보 리스트 반환


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
# 카메라를 찾고 정보를 출력하는 함수 정의 (필터 옵션 포함) 
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []
    # 모든 카메라 정보를 저장할 리스트 생성
    if camera_type_filter:
    # 카메라 타입 필터가 지정되었으면
        camera_type_filter = camera_type_filter.lower()
        # 소문자로 변환 (대소문자 구분 없이)

    if camera_type_filter is None or camera_type_filter == "opencv":
    # 필터가 없거나 "opencv"면
        all_cameras_info.extend(find_all_opencv_cameras())
        # OpenCV 카메라 정보를 전체 리스트에 추가
    if camera_type_filter is None or camera_type_filter == "realsense":
    # 필터가 없거나 "realsense"면
        all_cameras_info.extend(find_all_realsense_cameras())
        # RealSense 카메라 정보를 전체 리스트에 추가

    if not all_cameras_info:
    # 카메라를 하나도 찾지 못했으면
        if camera_type_filter:
        # 특정 타입으로 필터링했는데 못 찾았으면
            logger.warning(f"No {camera_type_filter} cameras were detected.")
            # 해당 타입 카메라 없음 경고
        else:
        # 필터 없이 검색했는데 못 찾았으면
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
            # 모든 타입 카메라 없음 경고
    else:
    # 카메라를 찾았으면
        print("\n--- Detected Cameras ---")
        # 카메라 목록 출력 시작 헤더
        for i, cam_info in enumerate(all_cameras_info):
        # 찾은 각 카메라에 대해 번호와 함께
            print(f"Camera #{i}:")
            # 카메라 번호 출력
            for key, value in cam_info.items():
            # 카메라 정보의 각 항목에 대해
                if key == "default_stream_profile" and isinstance(value, dict):
                # 기본 스트림 프로필이고 딕셔너리면
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    # 키 이름을 보기 좋게 변환해서 출력
                    for sub_key, sub_value in value.items():
                    # 스트림 프로필의 각 세부 항목에 대해
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                        # 세부 항목 출력 (Format, Width, Height, Fps 등)
                else:
                # 일반적인 정보면
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
                    # 키-값 쌍으로 출력 (Name, Type, Id 등)
            print("-" * 20)
            # 카메라 간 구분선 출력
    return all_cameras_info
    # 모든 카메라 정보 반환


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
):
# 단일 이미지를 디스크에 저장하는 함수
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
    # 예외 처리 시작
        img = Image.fromarray(img_array, mode="RGB")
        # NumPy 배열을 PIL 이미지로 변환 (RGB 모드)
        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        # 카메라 식별자에서 파일명에 사용할 수 없는 문자들을 안전한 문자로 치환
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        # 파일명 접두사 생성 (예: "opencv_0", "realsense_123456")
        filename = f"{filename_prefix}.png"
        # 완전한 파일명 생성
        path = images_dir / filename
        # 전체 파일 경로 생성
        path.parent.mkdir(parents=True, exist_ok=True)
        # 필요하면 디렉토리 생성
        img.save(str(path))
        # 이미지를 PNG 파일로 저장
        logger.info(f"Saved image: {path}")
        # 저장 완료 로그
    except Exception as e:
    # 예외 발생 시
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")
        # 저장 실패 오류 로그

def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
# 카메라 메타데이터를 바탕으로 카메라 인스턴스를 생성하는 함수
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    # 카메라 타입 추출 ("OpenCV" 또는 "RealSense")
    cam_id = cam_meta.get("id")
    # 카메라 ID 추출 (숫자 또는 시리얼 번호)
    instance = None
    # 인스턴스 변수 초기화
    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")
    # 카메라 준비 로그
    try:
    # 예외 처리 시작
        if cam_type == "OpenCV":
        # OpenCV 카메라면
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            # OpenCV 카메라 설정 생성 (RGB 모드)
            instance = OpenCVCamera(cv_config)
            # OpenCV 카메라 인스턴스 생성
        elif cam_type == "RealSense":
        # RealSense 카메라면
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            # RealSense 카메라 설정 생성
            instance = RealSenseCamera(rs_config)
            # RealSense 카메라 인스턴스 생성
        else:
        # 알 수 없는 카메라 타입이면
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            # 경고 로그 출력
            return None

        if instance:
        # 인스턴스가 생성되었으면
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            # 연결 시도 로그
            instance.connect(warmup=False)
            # 카메라에 연결 (워밍업 없이)
            return {"instance": instance, "meta": cam_meta}
            # 인스턴스와 메타데이터를 딕셔너리로 반환
    except Exception as e:
    # 예외 발생 시
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        # 연결 실패 오류 로그
        if instance and instance.is_connected:
        # 인스턴스가 있고 연결되어 있으면
            instance.disconnect()
            # 연결 해제
        return None


def process_camera_image(
    cam_dict: dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
# 단일 카메라에서 이미지를 캡처하고 처리하는 함수
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    # 카메라 인스턴스 추출
    meta = cam_dict["meta"]
    # 카메라 메타데이터 추출
    cam_type_str = str(meta.get("type", "unknown"))
    # 카메라 타입을 문자열로 변환
    cam_id_str = str(meta.get("id", "unknown"))
    # 카메라 ID를 문자열로 변환

    try:
    # 예외 처리 시작
        image_data = cam.read()
        # 카메라에서 이미지 데이터 읽기
        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
        # 읽은 이미지를 저장하고 결과 반환
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
# 모든 카메라 연결을 해제하는 함수
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
):
# 모든 카메라에서 이미지를 저장하는 메인 함수
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                            If None, uses all detected cameras.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # 출력 디렉토리 생성
    logger.info(f"Saving images to {output_dir}")
    # 저장 위치 로그
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)
    # 카메라 찾기 및 정보 출력
    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []
                current_capture_time = time.perf_counter()

                for cam_dict in cameras_to_use:
                    future = process_camera_image(cam_dict, output_dir, current_capture_time)
                    if future:
                        futures.append(future)

                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
        finally:
            print("\nFinalizing image saving...")
            executor.shutdown(wait=True)
            cleanup_cameras(cameras_to_use)
            print(f"Image capture finished. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
