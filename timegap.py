# coding: utf-8
import time


class TimeGap(object):
    def __init__(self):
        self.__start_time = -1
        self.__end_time   = -1


    def set_start_time(self, start: float):
        self.__start_time = start
        return self


    def set_end_time(self, end: float):
        self.__end_time = end

        if self.__end_time != -1 and self.__start_time > self.__end_time:
            self.__save_record()

        return self


    def time_gap(self, use_formated: bool = False) -> float:
        if self.__end_time == -1:
            gap = time.time() - self.__start_time
        else:
            gap = self.__end_time - self.__start_time

        if use_formated:
            return time.strftime("%Y-%m-%d %H:%M:%S", gap)
        return gap
        

    def start_time(self, use_formated: bool = False):
        if use_formated:
            return time.strftime("%Y-%m-%d %H:%M:%S", self.__start_time)
        return self.__start_time
    

    def end_time(self, use_formated: bool = False):
        if use_formated:
            return time.strftime("%Y-%m-%d %H:%M:%S", self.__end_time)
        return self.__end_time


    def __save_record(self):
        file_path = f"found-record-{time.strftime('%Y-%m-%d', time.time())}.log"
        with open(file_path, 'a', encoding="utf-8") as f:
            f.write(f'start time: {self.start_time(use_formated=True)}\n')
            f.write(f'end   time: {self.end_time(use_formated=True)}\n')
            f.write(f'gap   time: {self.time_gap(use_formated=True)}\n\n')
