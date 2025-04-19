# coding: utf-8
import os
import time
import subprocess

from logger import LOGGER


class Stream2Image(object):
    SI_ROOT_SAVE_DIRECTORY = "./webimages"

    SI_SAMPLING_GAP_MINUTE = 10


    @staticmethod
    def get_image(stream_url: str, organization_name: str, device_id: str) -> str:
        organization_path = os.path.join(Stream2Image.SI_ROOT_SAVE_DIRECTORY, organization_name)
        if not os.path.exists(organization_path):
            os.makedirs(organization_path)

        device_path = os.path.join(organization_path, device_id)
        if not os.path.exists(device_path):
            os.makedirs(device_path)

        file_path = os.path.join(device_path, f"{time.strftime('%Y-%m-%d_%H-%M-%M', time.localtime())}.jpg")

        process_command = [
            'ffmpeg',
            '-i', stream_url,
            '-ss', '00:00:05',
            '-vframes', '1',
            '-q:v', '2',
            file_path
        ]

        try:
            gotten_flag = False
            for _ in range(10):
                # subprocess.run(process_command, stdout=subprocess.PIPE)
                subprocess.run(process_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(file_path):
                    gotten_flag = True
                    LOGGER.info(f"Gotten an image of {organization_name}-{device_id}, saved to {file_path}")
                    break
                time.sleep(3)
            if not gotten_flag:
                LOGGER.error(f"Failed to get the image of {organization_name}-{device_id} after 10 attempts")
                return None
        except Exception as e:
            LOGGER.error(e)
            return None
        
        return file_path
