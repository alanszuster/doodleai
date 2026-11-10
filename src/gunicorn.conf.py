"""Gunicorn settings.

One worker on purpose: the rate limiter keeps its counters in memory, so more
workers would each get their own budget. TensorFlow is not fork-safe, so the model
is loaded in the worker (no ``preload_app``) and requests are handled one at a time.
"""

bind = "0.0.0.0:8080"
workers = 1
threads = 1
timeout = 30
graceful_timeout = 20
keepalive = 5
limit_request_line = 4094
limit_request_fields = 50
accesslog = None  # the app logs every request itself (with request id and duration)
errorlog = "-"
