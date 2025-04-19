# coding: utf-8
import queue
import threading


IMAGE_FILES_QUEUE_MAP = {
    "default": queue.Queue()
}

IMAGE_FILES_QUEUE_LOCK = threading.Lock()
