from datetime import datetime

def second2timestamp(second):
    hours, remainder = divmod(second, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return time_str

def is_hhmmss(time_str: str) -> bool:
    try:
        datetime.strptime(time_str, "%H:%M:%S")
        return True
    except ValueError:
        return False