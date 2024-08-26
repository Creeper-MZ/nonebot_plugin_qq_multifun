# logger.py
import time

def log_with_timestamp(message):
    print(f"[{time.time():.3f}] {message}")