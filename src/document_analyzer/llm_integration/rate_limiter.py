import time
from collections import deque
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential

class RateLimiter:
    def __init__(self, max_requests: int, per_seconds: int):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = deque()

    def __call__(self, func):
        @wraps(func)
        @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
        def wrapper(*args, **kwargs):
            while self.requests and self.requests[0] < time.time() - self.per_seconds:
                self.requests.popleft()

            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.per_seconds - time.time()
                if sleep_time > 0:
                    print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)

            self.requests.append(time.time())
            return func(*args, **kwargs)
        return wrapper
