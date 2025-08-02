import time
from collections import deque
from ..core.config import settings

class RateLimiter:
    def __init__(self, max_requests: int = None, per_seconds: int = 60):
        self.max_requests = max_requests or settings.llm.rate_limit
        self.per_seconds = per_seconds
        self.requests = deque()

    def __call__(self):
        while self.requests and self.requests[0] < time.time() - self.per_seconds:
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.per_seconds - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.requests.append(time.time())
