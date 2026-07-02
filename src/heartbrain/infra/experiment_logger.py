"""Experiment tracking utility.

Each run is stored in a numbered subdirectory (e.g. ``000/``, ``001/``, ...). 
A CSV index tracks all runs of the same type, with one column per parameter.

If this module changes location in the project (or elsewhere), please check
if the project root directory is still valid.
"""
import csv
import os
from contextlib import contextmanager
from datetime import datetime
from typing import TextIO


# Constants
INDEX_FILENAME = "index.csv"
LOG_FILENAME = "log.txt"

# Project root: three levels up from this file.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_MODULE_DIR)))
_DEFAULT_BASE_DIR = os.path.join(_PROJECT_ROOT, "output")


class ExperimentLogger:
    """Allocates a run directory and updates the index for one experiment run."""

    def __init__(self,
                 experiment_type: str,
                 parameters: dict,
                 component: str | None = None,
                 base_dir: str = _DEFAULT_BASE_DIR):
        """Configure the logger. No I/O is performed here.

        Parameters
        ----------
        experiment_type : str
            Name of the experiment family (e.g. "gain_sweep").
        parameters : dict
            Parameters of this run. Every key becomes a column in the index.
        component : str, optional
            Optional grouping level above ``experiment_type`` (e.g. "brain").
        base_dir : str
            Root directory for all experiments. Defaults to <project_root>/output.
        """
        self._experiment_type = experiment_type
        self._parameters = parameters
        self._component = component
        self._base_dir = base_dir

    @property
    def experiment_dir(self) -> str:
        """Directory containing all runs of this experiment type."""
        parts = [self._base_dir]
        if self._component is not None:
            parts.append(self._component)
        parts.append(self._experiment_type)
        return os.path.join(*parts)

    def _allocate_run_id(self) -> str:
        """Return the next available run ID by inspecting existing directories."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        existing = [
            entry for entry in os.listdir(self.experiment_dir)
            if entry.isdigit() and os.path.isdir(os.path.join(self.experiment_dir, entry))
        ]
        next_id = max((int(e) for e in existing), default=-1) + 1
        return f"{next_id:03d}"

    def _update_index(self, run_id: str) -> None:
        """Append a row to index.csv, creating the file with headers if needed."""
        index_path = os.path.join(self.experiment_dir, INDEX_FILENAME)
        columns = ["run_id", "timestamp", *self._parameters.keys(), "notes"]
        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": "",
            **self._parameters,
        }

        write_headers = not os.path.exists(index_path)
        with open(index_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(row)

    @contextmanager
    def open(self):
        """Open the log file for writing. Yields a file handle usable with ``print``.

        Creates the run directory and updates the index automatically.
        """
        run_id = self._allocate_run_id()
        run_dir = os.path.join(self.experiment_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        log_path = os.path.join(run_dir, LOG_FILENAME)
        f: TextIO = open(log_path, "w")
        try:
            yield f
        finally:
            f.close()
            self._update_index(run_id)