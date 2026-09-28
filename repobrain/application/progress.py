from contextvars import ContextVar

# Per-job callback; concurrent API operations cannot overwrite another job's progress.
progress_listener = ContextVar('indexing_progress', default=None)
