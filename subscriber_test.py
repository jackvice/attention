# subscriber_test.py
from multiprocessing.managers import BaseManager
import time
import numpy as np

# Interface declarations (no callable=)
class BufferManager(BaseManager): pass

BufferManager.register("get_buffer")
BufferManager.register("get_lock")

# Connect to the server running inside ros2_camera.py
mgr = BufferManager(address=("localhost", 50055), authkey=b"secret")
mgr.connect()

shared_buffer = mgr.get_buffer()
lock = mgr.get_lock()

while True:
    lock.acquire()
    try:
        if len(shared_buffer) == 5:
            print(f"[Subscriber] Received {len(shared_buffer)} frames")
            print(f"Mean pixel of latest frame: {np.mean(shared_buffer[-1]):.2f}")
        else:
            print(f"[Subscriber] Buffer incomplete ({len(shared_buffer)} frames)")
    finally:
        lock.release()
    time.sleep(0.5)




exit()
# subscriber_test.py
from multiprocessing.managers import BaseManager, SyncManager
import time
import numpy as np

class BufferManager(SyncManager): pass

BufferManager.register(
    "get_buffer",
    callable=lambda: BUFFER,
    exposed=['__getitem__', '__setitem__', '__delitem__', '__len__', 'append', 'extend', 'clear', '__iter__']
)
BufferManager.register(
    "get_lock",
    callable=lambda: BUFFER_LOCK,
    exposed=['acquire', 'release']
)



# Connect to the running manager (started inside ros2_camera.py)
mgr = BufferManager(address=("localhost", 50055), authkey=b"secret")
mgr.connect()

shared_buffer = mgr.get_buffer()
lock = mgr.get_lock()

while True:
    lock.acquire()
    try:
        if len(shared_buffer) == 5:
            print(f"[Subscriber] Received {len(shared_buffer)} frames")
            print(f"Mean pixel of latest frame: {np.mean(shared_buffer[-1]):.2f}")
        else:
            print(f"[Subscriber] Buffer incomplete ({len(shared_buffer)} frames)")
    finally:
        lock.release()
    time.sleep(0.5)


