# coding: utf-8
import os
import time
import ffmpeg
import threading
from typing import TypeAlias

from logger import LOGGER
from videostream.stream import ChinaMobileStream
from tasks.utils import check_hour_in_range


_Second: TypeAlias = int


class VideoRecorder(threading.Thread):
    def __init__(
            self, 
            source_name: str,
            device_name: str,
            duration: _Second = 2 * 60 * 60,
            start_hour: int = None,
            video_path: str = "./webvideos",
        ):
        super().__init__()

        self.source_name = source_name
        self.device_name = device_name

        assert duration > 0, "Duration must be greater than 0."
        # hours, seconds = divmod(duration, 3600)
        # minutes, seconds = divmod(seconds, 60)
        # self.duration = f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"
        self.duration = duration
        
        assert start_hour is None or 0 <= start_hour <= 23, "Start hour must be between 0 and 23."
        self.start_hour = time.localtime().tm_hour if start_hour is None else start_hour
        hours, _ = divmod(duration, 3600)
        self.end_hour = (self.start_hour + hours + 1) % 24

        video_path = os.path.join(video_path, source_name)
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        file_name = f"{time.strftime('%Y%m%d', time.localtime())}_{device_name}.mp4"
        self.video_file = os.path.join(video_path, file_name)

        self.recorded_flag = False

        self.name = f"{device_name}-recorder"


    def join(self):
        super().join()
        if self.recorded_flag:
            LOGGER.info(f"Video saved to {self.video_file}")
            return self.video_file


    def run(self):
        while True:
            current_hour = time.localtime().tm_hour
            if check_hour_in_range(current_hour, self.start_hour, self.end_hour):
                break
            else:
                time.sleep(5 * 60)
        
        LOGGER.info(f"Video recording started.")
        try:
            stream_url = ChinaMobileStream().get_device_play_url(self.device_name)[self.device_name]["hlsUrl"]
            (
                ffmpeg
                .input(stream_url, t=self.duration)
                .output(self.video_file, c="copy", vsync="1")
                .global_args('-loglevel', 'error')
                .run(overwrite_output=True, capture_stdout=False, capture_stderr=True)
            )
            self.recorded_flag = True
        except Exception or ffmpeg.Error as e:
            LOGGER.error(f"Video recording failed: {e}")


def test_run(duration: _Second = 1 * 60):
    devices_map = {
        "昆明黄冈实验学校": [
            "QLY-68ee4b19b0b8",
            "QLY-68ee4b19ba8b",
            "QLY-68ee4b19d26d",
            "QLY-68ee4b19d31e",
            "QLY-68ee4b19f112",
            "QLY-68ee4b1f50de",
            "QLY-68ee4b1fe44a",
            "QLY-fc5f4923d181",
        ],
    }

    tasks: list[VideoRecorder] = list()
    for source_name in devices_map.keys():
        for device_id in devices_map[source_name]:
            recorder = VideoRecorder(
                source_name=source_name,
                device_name=device_id,
                video_path="./webvideos",
                duration=duration,
                start_hour=0
            )
            recorder.setName(recorder.name)
            tasks.append(recorder)
            recorder.start()

    for task in tasks:
        task.join()
