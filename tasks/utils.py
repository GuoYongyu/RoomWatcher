# coding: utf-8


def check_hour_in_range(now_hour: int, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return start_hour == now_hour
    elif start_hour < end_hour:
        return now_hour >= start_hour and now_hour < end_hour
    else:
        return now_hour >= start_hour or now_hour < end_hour
