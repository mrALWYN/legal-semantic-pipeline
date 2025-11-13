# app/services/system_monitor.py
import psutil
import asyncio
import logging
from app.services.metrics import cpu_usage_percent, memory_usage_bytes

logger = logging.getLogger(__name__)

async def collect_system_metrics():
    """Collect system resource metrics periodically"""
    logger.info("Starting system metrics collection")
    while True:
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_usage_percent.set(cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_bytes.set(memory.used)
            
            logger.debug(f"System metrics collected: CPU={cpu_usage}%, Memory={memory.used} bytes")
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        # Wait before next collection
        await asyncio.sleep(30)  # Collect every 30 seconds
