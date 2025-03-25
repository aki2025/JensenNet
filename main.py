import asyncio
import signal
from applications.network_optimization_app import NetworkOptimizer
from applications.predictive_maintenance_app import MaintenancePredictor
from config import THRESHOLD_PATH
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_all():
    try:
        with open(THRESHOLD_PATH, "r") as f:
            threshold = float(f.read())
        optimizer = NetworkOptimizer(threshold)
        predictor = MaintenancePredictor()
        await asyncio.gather(optimizer.optimize(), predictor.predict())
    except Exception as e:
        logger.error(f"Error in main loop: {e}")

def shutdown_handler(loop):
    logger.info("Shutting down...")
    for task in asyncio.all_tasks(loop):
        task.cancel()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown_handler, loop)
    try:
        loop.run_until_complete(run_all())
    except asyncio.CancelledError:
        pass
    finally:
        loop.close() 
