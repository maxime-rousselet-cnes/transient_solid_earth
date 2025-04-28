"""
Real time file creation observer. Usefull for adaptative step parallel computin loop.
"""

import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .paths import INTERMEDIATE_RESULT_STRING


class FileCreationObserver:
    """
    Monitors a directory for new file creations.
    """

    observer: BaseObserver
    new_file_detected: threading.Event
    base_path: Path
    created_file_paths: list[Path]

    def __init__(self, base_path: Path):
        """
        Initializes the file creation observer and starts it immediately.
        """

        base_path.mkdir(parents=True, exist_ok=True)

        self.observer: BaseObserver = Observer()
        self.new_file_detected: threading.Event = threading.Event()
        self.base_path: Path = base_path
        self.created_file_paths = []

        event_handler = self.Handler(
            event_flag=self.new_file_detected, created_file_paths=self.created_file_paths
        )
        self.observer.schedule(event_handler, self.base_path, recursive=True)
        self.observer.start()

    class Handler(FileSystemEventHandler):
        """
        Handler for file system events. Sets an event flag when a new file is created.
        """

        event_flag: threading.Event
        created_file_paths: list[Path]

        def __init__(self, event_flag: threading.Event, created_file_paths: list[Path]):
            """
            Initializes the handler with an event flag.
            """

            self.event_flag = event_flag
            self.created_file_paths = created_file_paths

        def on_created(self, event):
            """
            Sets the event flag if a file (not directory) is created.
            """

            if not event.is_directory:
                self.event_flag.set()
                self.created_file_paths.append(Path(event.src_path))

    def file_has_been_created(self) -> bool:
        """
        Checks if a new file has been created since the last check.
        """

        if self.new_file_detected.is_set():
            self.new_file_detected.clear()
            return True
        return False

    def get_created_file_paths(self) -> list[Path]:
        """
        Returns the list of paths of all files that have been created so far.
        """

        created_file_paths = self.created_file_paths.copy()
        self.created_file_paths.clear()
        return [
            file_path.parent
            for file_path in created_file_paths
            if "imag" in file_path.name  # Because "real" file is always created first.
            and file_path.parent.parent.parent.parent.name == INTERMEDIATE_RESULT_STRING
        ]

    def stop(self) -> None:
        """
        Stops the observer and cleans up resources.
        """

        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
