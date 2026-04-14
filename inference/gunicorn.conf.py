from prometheus_client import multiprocess

workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = 120
accesslog = "-"

def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)