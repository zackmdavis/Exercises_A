# http://reddit.com
# /r/dailyprogrammer/comments/yqydh/8242012_challenge_91_easy_sleep_sort/

import threading
import time

# it feels pretty trivial in Python, though, of course, needless to
# say

TICK = 10
SECOND_TO_TICK_SCALING_FACTOR = TICK/1000

def thread_task(interval):
    time.sleep(interval * SECOND_TO_TICK_SCALING_FACTOR)
    print(interval, end=' ')

def printing_sleepsort(intervals):
    tasks = [
        threading.Thread(
            target=thread_task,
            args=(interval,)
        )
        for interval in intervals
    ]
    for task in tasks:
        task.start()
    while any(task.is_alive() for task in tasks):
        time.sleep(0.5)
    print()  # final newline

if __name__ == "__main__":
    printing_sleepsort([5, 2, 7, 4, 6, 3, 8, 1])
