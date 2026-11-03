"""PTY session: spawn the drcode bundle inside a pseudo-terminal.

DrCode Desktop is byte-level plumbing only: it owns a PTY, starts the
existing drcode payload with ``cwd`` = the user's chosen project, and
relays bytes both ways. Zero drcode logic lives here — functional parity
comes from running the unmodified bundle.

The child gets TERM=xterm-256color / COLORTERM=truecolor so Textual
enables its full-render path; SIGWINCH handling (window resize) is
Textual's own — we only push the new size with ioctl(TIOCSWINSZ).
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import struct
import subprocess
import sys
import termios
from collections.abc import Callable
from pathlib import Path

# Forced child environment. Everything else is inherited from the shell
# that launched the app (DISPLAY/WAYLAND_DISPLAY, SSH_AUTH_SOCK, PATH...).
_CHILD_ENV_OVERRIDES: dict[str, str] = {
    "TERM": "xterm-256color",
    "COLORTERM": "truecolor",
}

# The PTY's line discipline strips these before the program sees them;
# make sure a stale value from the parent GUI session never leaks in.
_CHILD_ENV_UNSET = ("COLUMNS", "LINES")

READ_CHUNK = 65536


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for the spawned terminal program."""
    env = dict(os.environ)
    for key in _CHILD_ENV_UNSET:
        env.pop(key, None)
    env.update(_CHILD_ENV_OVERRIDES)
    if not env.get("LANG"):
        env["LANG"] = "en_US.UTF-8"
    if extra:
        env.update(extra)
    return env


def _claim_controlling_tty(slave_fd: int) -> Callable[[], None]:
    """preexec_fn: make the slave pty the child's controlling terminal.

    subprocess does setsid() for us (start_new_session=True); a session
    leader still needs an explicit TIOCSCTTY to *acquire* the ctty, which
    job control (Ctrl+C, Ctrl+Z inside the TUI) relies on.
    """

    def fn() -> None:  # pragma: no cover - runs in the forked child
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        except OSError:
            pass  # best effort: some platforms grant the ctty implicitly

    return fn


class PtySession:
    """One terminal program behind one PTY, pumped by asyncio.

    ``on_output(bytes)`` is called from the event loop as raw PTY output
    arrives; ``on_exit(exit_code)`` exactly once when the child is reaped.
    """

    def __init__(
        self,
        argv: list[str],
        cwd: str | Path,
        *,
        on_output: Callable[[bytes], None],
        on_exit: Callable[[int], None],
        env_extra: dict[str, str] | None = None,
    ) -> None:
        self.argv = argv
        self.cwd = str(cwd)
        self._on_output = on_output
        self._on_exit = on_exit
        self._env = child_environment(env_extra)

        self.master_fd: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._write_buffer = bytearray()
        self._writer_attached = False
        self._reader_registered = False
        self._reaper_task: asyncio.Task[None] | None = None
        self._exit_code: int | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self, *, initial_size: tuple[int, int] | None = None) -> None:
        """Open the PTY and spawn the child. Must run inside a running loop."""
        self._loop = asyncio.get_running_loop()
        master_fd, slave_fd = os.openpty()
        if initial_size is not None:
            self._ioctl_winsz(master_fd, *initial_size)

        self.master_fd = master_fd
        # PLW1509: preexec_fn is the only way to claim the controlling
        # terminal before exec; the forked child does nothing except a
        # single ioctl, so thread-unsafety is not a concern.
        # ASYNC220: the spawn is fast and one-shot per session.
        self._proc = subprocess.Popen(  # noqa: ASYNC220
            self.argv,
            cwd=self.cwd,
            env=self._env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
            preexec_fn=_claim_controlling_tty(slave_fd),  # noqa: PLW1509
        )
        os.close(slave_fd)  # parent keeps only the master

        self._loop.add_reader(master_fd, self._on_readable)
        self._reader_registered = True
        self._reaper_task = self._loop.create_task(self._reap())

    async def _reap(self) -> None:
        assert self._proc is not None
        code = await asyncio.to_thread(self._proc.wait)
        self._exit_code = code
        self._teardown_fds()
        self._on_exit(code)

    def _teardown_fds(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None or self.master_fd is None:
            return
        if self._reader_registered:
            try:
                loop.remove_reader(self.master_fd)
            except (OSError, ValueError):
                pass
            self._reader_registered = False
        if self._writer_attached:
            try:
                loop.remove_writer(self.master_fd)
            except (OSError, ValueError):
                pass
            self._writer_attached = False
        try:
            os.close(self.master_fd)
        except OSError:
            pass

    # -- io ----------------------------------------------------------------

    def _on_readable(self) -> None:
        assert self.master_fd is not None
        try:
            data = os.read(self.master_fd, READ_CHUNK)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            # EIO on Linux: child closed the slave side. The reaper will
            # deliver the exit event; drop the reader so we don't spin.
            if self._reader_registered and self._loop is not None:
                try:
                    self._loop.remove_reader(self.master_fd)
                except (OSError, ValueError):
                    pass
                self._reader_registered = False
            return
        if not data:
            return
        self._on_output(data)

    def write(self, data: bytes) -> None:
        """Queue bytes for the child (non-blocking; drained by the loop)."""
        if self.master_fd is None or self._closed or not data:
            return
        self._write_buffer.extend(data)
        self._flush()

    def _flush(self) -> None:
        loop = self._loop
        if loop is None or self.master_fd is None or self._closed:
            return
        while self._write_buffer:
            try:
                written = os.write(self.master_fd, self._write_buffer)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                # Child is gone or the PTY is dead; the reaper reports it.
                self._write_buffer.clear()
                return
            del self._write_buffer[:written]
        if self._write_buffer and not self._writer_attached:
            loop.add_writer(self.master_fd, self._flush)
            self._writer_attached = True
        elif not self._write_buffer and self._writer_attached:
            loop.remove_writer(self.master_fd)
            self._writer_attached = False

    # -- control -----------------------------------------------------------

    def resize(self, rows: int, cols: int) -> None:
        """Propagate a window resize; Textual reacts to the SIGWINCH itself."""
        if self.master_fd is None or self._closed or rows <= 0 or cols <= 0:
            return
        self._ioctl_winsz(self.master_fd, rows, cols)
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGWINCH)
            except ProcessLookupError:
                pass

    @staticmethod
    def _ioctl_winsz(fd: int, rows: int, cols: int) -> None:
        try:
            fcntl.ioctl(
                fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", rows, cols, 0, 0),
            )
        except OSError:
            pass

    def terminate(self, *, grace: float = 2.0) -> None:
        """Ask the child to quit; force-kill after ``grace`` seconds."""
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return

        def _kill_later() -> None:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass

        # Best-effort force-kill without blocking anything.
        timer = None
        loop = self._loop
        if loop is not None and loop.is_running():
            timer = loop.call_later(grace, _kill_later)
            if self._reaper_task is not None:
                self._reaper_task.add_done_callback(lambda _t: timer.cancel())
        else:
            _kill_later()

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def payload_argv(payload: str, args: list[str] | None = None) -> list[str]:
    """argv to run the drcode onedir bundle (no modifications to it)."""
    argv = [payload]
    if args:
        argv.extend(args)
    return argv


def resolve_payload() -> Path:
    """Locate the bundled drcode payload.

    Frozen app: <_MEIPASS>/resources/drcode/drcode (datas entry in
    desktop.spec). Dev mode: desktop/resources/drcode/drcode, then
    dist/drcode/drcode at the repo root.
    """
    exe = "drcode.exe" if sys.platform == "win32" else "drcode"
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "resources" / "drcode" / exe)
    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "resources" / "drcode" / exe,
            here.parent / "dist" / "drcode" / exe,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "drcode payload not found (build it with `bash scripts/build-binary.sh`); "
        f"looked in: {tried}"
    )
