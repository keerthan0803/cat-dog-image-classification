# Gunicorn configuration file
import multiprocessing
import os

# Bind to the port provided by Render
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker configuration
workers = 1  # Use only 1 worker to save memory
worker_class = "sync"
threads = 2  # Use threads instead of multiple workers

# Timeout settings
timeout = 120  # Increase timeout to 120 seconds for model loading and prediction
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory management
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10

# Preload app to share model across threads
preload_app = False  # Set to False for lazy loading
