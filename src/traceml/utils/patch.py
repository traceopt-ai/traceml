from queue import Queue


# Shared queues for parameter-memory and activation events
model_queue: Queue = Queue()


def get_model_queue() -> Queue:
    """Return the shared queue of models for parameter-memory sampling."""
    return model_queue
