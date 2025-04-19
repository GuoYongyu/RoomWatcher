# coding: utf-8
import queue

from frame.frame import Frame
from timegap import TimeGap


class SlideWindow:
    SLIDE_WINDOW_SIZE = 10


    def __init__(self, window_size: int = None):
        self.window_size = window_size if window_size is not None else SlideWindow.SLIDE_WINDOW_SIZE
        assert self.window_size > 1, 'window.SlideWindow: Window size must be greater than 1.'
        self.window = queue.Queue(maxsize=window_size)

        self.time_gap = TimeGap()


    def push(self, frame: Frame):
        if self.window.full():
            self.window.get()

        if frame.may_have_exception:
            if self.window.empty() or not self.window[-1].may_have_exception:
                self.time_gap.set_start_time(frame.timestamp)
        else:
            # if the last frame has mouse, set end time
            if self.window[-1].may_have_exception:
                self.time_gap.set_end_time(frame.timestamp)
                # update time gap to a new one
                self.time_gap = TimeGap()
        
        self.window.put(frame)


    def pop(self):
        if not self.window.empty():
            return self.window.get()
        else:
            return None
    

    def is_empty(self):
        return self.window.empty()
    

    def front(self):
        if not self.is_empty():
            return self.window.queue[0]
        else:
            return None
