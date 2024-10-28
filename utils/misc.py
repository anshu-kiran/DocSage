import time
import threading

def loading_animation(stop_event):
    animation = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\rGenerating... {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.1)

def start_animation():
    stop_event = threading.Event()
    thread = threading.Thread(target=loading_animation, args=(stop_event,))
    thread.start()
    return stop_event, thread

def stop_animation(stop_event, thread):
    stop_event.set()
    thread.join()