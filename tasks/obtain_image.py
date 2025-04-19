# coding: utf-8
import queue
import threading
import time

from videostream.stream import ChinaMobileStream
from videostream.stream2image import Stream2Image
from logger import LOGGER
from tasks.data_queue import IMAGE_FILES_QUEUE_MAP
from tasks.utils import check_hour_in_range


class ObtainImageTask(threading.Thread):
    def __init__(
            self,
            source_name: str,
            device_ids: list[str],
            start_hour: int = 22,
            end_hour: int = 4,
            use_data_queue: bool = False
        ):
        """
        use_data_queue: 是否使用数据队列，默认为 False，数据队列用于连接图像处理任务
        """
        super().__init__()

        self.source_name = source_name
        assert isinstance(device_ids, list), "device_ids must be list type."
        self.device_ids = device_ids
        self.name = f"ObtainImage-{source_name}"

        self.use_data_queue = use_data_queue
        if use_data_queue and IMAGE_FILES_QUEUE_MAP.get(source_name, None) is None:
            IMAGE_FILES_QUEUE_MAP[source_name] = queue.Queue()

        self.start_hour = start_hour
        self.end_hour = end_hour

        self.stream = ChinaMobileStream()


    def __get_play_urls(self) -> dict[str, str]:
        return self.stream.get_device_play_url(self.device_ids)
    

    def join(self):
        super().join()
    

    def run(self):
        while True:
            now_time_hour = time.localtime().tm_hour
            
            if check_hour_in_range(now_time_hour, self.start_hour, self.end_hour):
                image_files = list()
                play_urls = self.__get_play_urls()
                for device_id, url in play_urls.items():
                    image = Stream2Image.get_image(url["hlsUrl"], self.source_name, device_id)
                    if image is not None:
                        image_files.append((device_id, image))
                
                if self.use_data_queue:
                    image_files = dict(sorted(image_files, key=lambda x: x[0]))
                    IMAGE_FILES_QUEUE_MAP[self.source_name].put(image_files)
                LOGGER.info(f'{self.source_name} has obtained {len(image_files)} images.')
                time.sleep(Stream2Image.SI_SAMPLING_GAP_MINUTE * 60)
            else:
                time.sleep(1 * 60)


def test_run(start_hour: int = None, end_hour: int = None):
    device_infos = {
        "昆明市一职中": [
            "LOCAL-129",
            "LOCAL-130",
            "LOCAL-131",
            "LOCAL-132",
            "LOCAL-133",
            "LOCAL-134",
            "LOCAL-135",
            "LOCAL-140",
            "LOCAL-141",
            "LOCAL-142",
            "LOCAL-143",
            "LOCAL-144",
        ],
        "呈贡区马郎小学": [
            "QLY-3ce36b9064e2",
            "QLY-3ce36b9064ef",
            "QLY-3ce36b9064ff",
            "QLY-3ce36b906734",
        ],
        "呈贡七甸中心学校七甸小学": [
            "QLY-743fc2a0f6dc",
            "QLY-743fc2a0f789",
            "QLY-743fc2a0f794",
        ],
        "云南民族大学附属高级中学（呈贡校区）": [
            "QLY-40471045881318156796",
            "QLY-40471045881318156797",
            "QLY-40471045881318156798",
            "QLY-40471045881318156799",
            "QLY-40471045881318156800",
            "QLY-40471045881318156801",
            "QLY-40471045881318156802",
            "QLY-40471045881318156803",
            "QLY-40471045881318156804",
            "QLY-40471045881318156805",
            "QLY-40471045881318156806",
            "QLY-40471045881318156807",
            "QLY-40471045881318156808",
            "QLY-40471045881318156809",
            "QLY-40471045881318156810",
        ],
        "昆明市官渡区第六中学": [
            "QLY-0403126e81b9",
            "QLY-0403126e831b",
            "QLY-80beafd9c54c",
            "QLY-80beafd9c553",
            "QLY-80beafd9c554",
            "QLY-80beafd9c55a",
            "QLY-80beafd9c55d",
            "QLY-80beafd9c564",
            "QLY-80beafd9c569",
            "QLY-80beafd9c6d7",
        ],
        "富民县东村中心小学": [
            "QLY-40471045881318472390",
            "QLY-40471045881318472391",
            "QLY-40471045881318472392",
            "QLY-40471045881318472393",
            "QLY-40471045881318472394",
            "QLY-40471045881318472395",
            "QLY-40471045881318472396",
            "QLY-40471045881318472397",
            "QLY-40471045881318472398",
            "QLY-40471045881318472399",
            "QLY-40471045881318472400",
            "QLY-40471045881318472401",
            "QLY-40471045881318472402",
            "QLY-40471045881318472403",
            "QLY-40471045881318472404",
            "QLY-40471045881318472405",
            "QLY-40471045881318472406",
            "QLY-40471045881318472407",
            "QLY-40471045881318472408",
        ],
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
        "寻甸县羊街镇康乐幼儿园": [
            "QLY-68ee4b1f6d80",
            "QLY-68ee4b1f6d89",
        ],
    }

    start_hour = 22 if start_hour is None else start_hour
    end_hour = 4 if end_hour is None else end_hour

    try:
        task_list: list[ObtainImageTask] = list()
        for name, ids in device_infos.items():
            task = ObtainImageTask(
                source_name=name, device_ids=ids, use_data_queue=False,
                start_hour=start_hour, end_hour=end_hour
            )
            task_list.append(task)

        for task in task_list:
            task.start()

        for task in task_list:
            task.join()
    except Exception as e:
        LOGGER.error(f"{e}")
