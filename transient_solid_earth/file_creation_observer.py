"""
Real-time file creation observer. Useful for adaptive step parallel computing loops.
"""

import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .paths import INTERMEDIATE_RESULT_STRING


class Handler(FileSystemEventHandler):
    """
    Handler for file system events. Sets an event flag when a new file is created or touched.
    """

    def __init__(self, event_flag: threading.Event, created_file_paths: list[Path]):
        self.event_flag = event_flag
        self.created_file_paths = created_file_paths
        self._seen = set()

    def _handle_event(self, path: str):
        p = Path(path)
        if p not in self._seen and not p.is_dir():
            self._seen.add(p)
            self.event_flag.set()
            self.created_file_paths.append(p)

    def on_created(self, event):
        self._handle_event(event.src_path)

    def on_modified(self, event):
        self._handle_event(event.src_path)

    def reset_seen(self):
        """
        Resets.
        """

        self._seen.clear()


class FileCreationObserver:
    """
    Monitors a directory for new file creations or touches.
    """

    observer: BaseObserver
    new_file_detected: threading.Event
    base_path: Path
    created_file_paths: list[Path]
    handler: Handler

    def __init__(self, base_path: Path):
        """
        Initializes the file creation observer and starts it immediately.
        """

        base_path.mkdir(parents=True, exist_ok=True)

        self.observer = Observer()
        self.new_file_detected = threading.Event()
        self.base_path = base_path
        self.created_file_paths = []

        self.handler = Handler(
            event_flag=self.new_file_detected, created_file_paths=self.created_file_paths
        )

        self.observer.schedule(self.handler, self.base_path, recursive=True)
        self.observer.start()

    def file_has_been_created(self) -> bool:
        """
        Checks if a new file has been created or touched since the last check.
        """

        if self.new_file_detected.is_set():
            self.new_file_detected.clear()
            return True
        return False

    def get_created_file_paths(self) -> list[Path]:
        """
        Returns the list of paths of all files that have been created or touched so far.
        """

        created_file_paths = self.created_file_paths.copy()
        self.created_file_paths.clear()
        self.handler.reset_seen()

        return [
            file_path.parent
            for file_path in created_file_paths
            if "imag" in file_path.name
            and file_path.parent.parent.parent.parent.name == INTERMEDIATE_RESULT_STRING
        ]

    def stop(self) -> None:
        """
        Stops the observer and cleans up resources.
        """

        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
