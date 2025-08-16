from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[{name}] {dt:.2f}s")

