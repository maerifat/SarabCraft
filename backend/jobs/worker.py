import logging
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.jobs.core import (
    append_job_event,
    claim_job,
    dequeue_job,
    get_job,
    initialize_runtime,
)
from backend.jobs.handlers import run_job
from utils.credentials import apply_credentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("mlsec.jobs.worker")


def worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    return f"{host}:{pid}"


def main() -> None:
    wid = worker_id()
    logger.info("Starting SarabCraft worker %s", wid)

    while True:
        try:
            apply_credentials()
            initialize_runtime()
            break
        except Exception:
            logger.exception("Failed to initialize worker dependencies, retrying")
            time.sleep(5)

    while True:
        try:
            job_id = dequeue_job(timeout=5)
            if not job_id:
                continue

            claimed = claim_job(job_id, wid)
            if not claimed:
                logger.info("Skipped unclaimable job %s", job_id)
                continue

            append_job_event(job_id, "progress", {"message": "Worker claimed job", "worker_id": wid})
            apply_credentials()
            job = get_job(job_id)
            if not job:
                logger.warning("Job disappeared after claim: %s", job_id)
                continue

            run_job(job)
        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down")
            raise
        except Exception:
            logger.exception("Unhandled worker loop error")
            time.sleep(2)


if __name__ == "__main__":
    main()
