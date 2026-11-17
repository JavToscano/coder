import io

from rich.console import Console

from coder.menu import select_index


def _keys(*ks):
    it = iter(ks)
    return lambda: next(it)


def _con():
    return Console(file=io.StringIO(), force_terminal=True)


def test_arrows_move_and_enter_selects():
    assert select_index(_con(), 3, lambda c: str(c), read_key=_keys("down", "down", "enter")) == 2


def test_wraps_around_and_ignores_unknown_keys():
    assert select_index(_con(), 3, lambda c: str(c), read_key=_keys("up", "", "enter")) == 2


def test_quit_returns_none():
    assert select_index(_con(), 3, lambda c: str(c), read_key=_keys("down", "quit")) is None
