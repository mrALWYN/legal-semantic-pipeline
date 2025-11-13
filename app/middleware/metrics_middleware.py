import time
from fastapi import Request
from app.services.metrics import api_requests_total, api_request_duration_seconds

async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Record request duration
    duration = time.time() - start_time
    api_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Record request count
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    return response
