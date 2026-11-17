"""Tiny arrow-key menu on top of rich.Live (no extra dependency; POSIX terminals only)."""
from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console, RenderableType
from rich.live import Live

UP, DOWN, ENTER, QUIT = "up", "down", "enter", "quit"


def _read_key() -> str:
    import os
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":  # arrow keys arrive as ESC [ A/B; a bare ESC is "cancel"
            if not select.select([fd], [], [], 0.05)[0]:
                return QUIT
            seq = os.read(fd, 2)
            return {b"[A": UP, b"[B": DOWN, b"OA": UP, b"OB": DOWN}.get(seq, "")
        if ch in (b"\r", b"\n"):
            return ENTER
        if ch in (b"k",):
            return UP
        if ch in (b"j",):
            return DOWN
        if ch in (b"q", b"\x03", b"\x04"):
            return QUIT
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_index(console: Console, count: int, render: Callable[[int], RenderableType], default: int = 0,
                 read_key: Callable[[], str] = _read_key) -> int | None:
    """Let the user move with ↑/↓ (or j/k) and press Enter. Returns the index, or None if cancelled."""
    cur = min(max(default, 0), count - 1)
    with Live(render(cur), console=console, auto_refresh=False, transient=False) as live:
        while True:
            key = read_key()
            if key == ENTER:
                return cur
            if key == QUIT:
                return None
            if key == UP:
                cur = (cur - 1) % count
            elif key == DOWN:
                cur = (cur + 1) % count
            else:
                continue
            live.update(render(cur), refresh=True)
