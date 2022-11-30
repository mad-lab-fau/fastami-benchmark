import signal

import resource


def get_free_memory() -> int:
    """Get free memory available on system in kB.

    Returns:
        Available memory in kB.
    """
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def get_total_memory() -> int:
    """Get total memory available on system in kB.

    Returns:
        Total memory in kB.
    """
    with open('/proc/meminfo', 'r') as mem:
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                return int(sline[1])

    return -1


def set_memory_limit(limit: int) -> None:
    """Set soft Memory Limit in Bytes.

    Args: 
        limit: Limit in Bytes.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,
                       (limit, hard))


class Timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        """Create a timeout context manager.

        Args:
            seconds: Timeout in seconds.
            error_message: Error message to raise in case of timeout.

        Raises:
            TimeoutError: If timeout is reached.
        """
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        """Handle timeout."""
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
