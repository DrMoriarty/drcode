"""``python -m desktop`` entry point (dev mode)."""

from .app import _exit_process, main

if __name__ == "__main__":
    # _exit_process, not SystemExit: pywebview's bridge threads can be left
    # blocked on a dead window (see desktop/app.py::_exit_process) and the
    # interpreter would join them forever.
    _exit_process(main())
