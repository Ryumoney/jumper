"""`mjrl.viewer.keys.from_viewer`: both edges of a key, out of a viewer that
reports its press alone.

The silent failure this file is for: MuJoCo's viewer calls `key_callback` on
`GLFW_PRESS` and on nothing else (`GlfwAdapter::IsKeyDownEvent`, 3.1 through
3.11), so a held key's auto-repeat and its release never reached Python. The
readers of the registry inferred the release from repeats, every key let go
0.75 s in, and a stick pushed from the keyboard stopped at 37.5% -- while every
test, written against repeats the viewer never sends, passed.

GLFW here is a fake with the two calls the hook makes. Each key event goes to
whatever callback the fake window holds at that moment, as GLFW's own dispatch
does, and MuJoCo's callback is modelled as 3.11 has it: the viewer's shortcuts
for every event, `key_callback` for a press only. On the real viewer the same
path was measured with synthetic X events: W held 1.5 s arrived as one call
before, and as one press and one release after.
"""

from __future__ import annotations

import ctypes

import pytest
from mjrl.viewer import keys

RELEASE, PRESS, REPEAT = 0, 1, 2
W = 87


class FakeGlfw:
    """One window, which is current, and whose key callback can be swapped."""

    def __init__(self, window: int = 0x1000) -> None:
        self.window = window
        self.callback: int | None = None

    def current(self) -> int:
        return self.window

    def set_key_callback(self, window: int, address: int) -> int | None:
        assert window == self.window
        previous, self.callback = self.callback, address
        return previous


class Viewer:
    """A window as MuJoCo sets one up: its own GLFW key callback, which runs
    the viewer's shortcuts -- counted here -- and calls `key_callback` on a
    press and on nothing else."""

    def __init__(self, glfw: FakeGlfw, key_callback) -> None:
        self.glfw = glfw
        self.shortcuts: list[tuple[int, int]] = []

        def on_key(window, key, scancode, action, mods):
            self.shortcuts.append((key, action))
            if action == PRESS:
                key_callback(key)

        self._on_key = keys._KEYFUN(on_key)  # GLFW holds a raw pointer to it
        glfw.callback = ctypes.cast(self._on_key, ctypes.c_void_p).value

    def event(self, key: int, action: int) -> None:
        """GLFW delivering one event to the callback the window holds now."""
        keys._KEYFUN(self.glfw.callback)(self.glfw.window, key, 0, action, 0)

    def hold(self, key: int, repeats: int) -> None:
        """A key held long enough for `repeats` auto-repeats, then let go."""
        self.event(key, PRESS)
        for _ in range(repeats):
            self.event(key, REPEAT)
        self.event(key, RELEASE)


@pytest.fixture
def fresh(monkeypatch):
    """The registry and the hook's state, emptied for one test; the edges its
    one handler sees."""
    glfw = FakeGlfw()
    monkeypatch.setattr(keys, "_glfw", glfw)
    monkeypatch.setattr(keys, "_previous", {})
    monkeypatch.setattr(keys, "_warned", False)
    monkeypatch.setattr(keys, "_handlers", [])
    monkeypatch.setattr(keys, "_reserved", frozenset())
    monkeypatch.delenv("MJRL_KEY_DEBUG", raising=False)
    seen: list[tuple[int, bool]] = []
    keys.register(lambda code, down: seen.append((code, down)))
    return glfw, seen


def test_a_held_key_is_one_press_and_one_release(fresh) -> None:
    """Held through thirty repeats and let go, twice: a press and a release
    each, and nothing for the repeats. MuJoCo's callback still sees every
    event, the second press included -- which reaches `from_viewer` only
    through the hook calling it first, so the viewer's shortcuts still fire.

    The control group is the wiring this replaced, `key_callback` straight into
    the registry as a press: the same holds give two presses and no release,
    which is the failure, so the fake viewer is one that can show it."""
    glfw, seen = fresh
    viewer = Viewer(glfw, keys.from_viewer)
    viewer.hold(W, repeats=30)
    viewer.hold(W, repeats=5)
    assert seen == [(W, True), (W, False), (W, True), (W, False)]
    assert len(viewer.shortcuts) == (1 + 30 + 1) + (1 + 5 + 1), "MuJoCo's callback was cut off"

    seen.clear()
    old = Viewer(FakeGlfw(), lambda code: keys.dispatch(code, True))
    old.hold(W, repeats=30)
    old.hold(W, repeats=5)
    assert seen == [(W, True), (W, True)], "the control group saw a release"


def test_a_window_opened_where_a_closed_one_was_is_hooked_again(fresh) -> None:
    """A second viewer in one process can get a window at the address the
    first one's had. The hook tells them apart by what GLFW hands back when it
    is set -- MuJoCo's callback on a new window -- rather than by the address,
    which would call the closed viewer's callback for the new one's keys. The
    first viewer's shortcut count, untouched by the second's events, is what
    shows which one was called."""
    glfw, seen = fresh
    first = Viewer(glfw, keys.from_viewer)
    first.hold(W, repeats=3)
    before = len(first.shortcuts)
    second = Viewer(glfw, keys.from_viewer)  # the same address, MuJoCo's callback again
    second.hold(W, repeats=3)
    assert seen == [(W, True), (W, False)] * 2
    assert len(first.shortcuts) == before, "the closed viewer's callback ran for the new one"
    assert len(second.shortcuts) == 1 + 3 + 1


def test_with_no_window_current_a_press_is_a_tap_and_it_says_so(fresh, capsys) -> None:
    """Without a window to hook, the release cannot be seen. A press is then a
    press and an immediate release -- a button clicks, a stick does not move --
    rather than a key stuck down, and the reason is printed once."""
    glfw, seen = fresh
    glfw.window = 0
    keys.from_viewer(W)
    keys.from_viewer(W)
    assert seen == [(W, True), (W, False)] * 2
    assert capsys.readouterr().out.count("each press is reported as a tap") == 1


UP = 265  # outside the printable range: no toggle-back


def _presses(viewer: Viewer, key: int) -> int:
    return sum(1 for k, a in viewer.shortcuts if k == key and a == PRESS)


def test_a_reserved_key_leaves_the_viewer_as_it_was(fresh) -> None:
    """`play --app` reserves the controller's keys, so D walking right stops
    also hiding every static geom. MuJoCo's shortcuts are toggles, so what
    shows whether one is left flipped is the parity of the presses its
    callback saw. The first press reaches MuJoCo before the hook exists; the
    hook hands it the key once more, so the count is even and the scene is as
    it was, while the handler still gets every edge.

    The control group is the same holds with nothing reserved: an odd count,
    the toggle left flipped, so the parity is a test that can fail."""
    glfw, seen = fresh
    keys.reserve({W, UP})
    viewer = Viewer(glfw, keys.from_viewer)
    viewer.hold(W, repeats=5)
    viewer.hold(W, repeats=5)
    viewer.hold(W, repeats=5)
    assert seen == [(W, True), (W, False)] * 3
    assert _presses(viewer, W) % 2 == 0, "a reserved key's shortcut was left toggled"
    assert _presses(viewer, W) == 2, "MuJoCo saw a reserved key after the hook was in"

    keys.reserve(())
    seen.clear()
    control = Viewer(FakeGlfw(), keys.from_viewer)
    control.hold(W, repeats=5)
    control.hold(W, repeats=5)
    control.hold(W, repeats=5)
    assert _presses(control, W) == 3, "the control group did not leave the toggle flipped"


def test_a_reserved_key_outside_the_toggles_is_not_pressed_twice(fresh) -> None:
    """The arrows' and Escape's shortcuts are not toggles, so a second press
    would do the thing twice rather than undo it. A reserved arrow as the very
    first press reaches MuJoCo once, unavoidably, and never again."""
    glfw, seen = fresh
    keys.reserve({W, UP})
    viewer = Viewer(glfw, keys.from_viewer)
    viewer.hold(UP, repeats=3)
    viewer.hold(UP, repeats=3)
    viewer.hold(W, repeats=3)
    assert _presses(viewer, UP) == 1
    assert _presses(viewer, W) == 0, "a reserved key reached MuJoCo once the hook was in"
    assert seen == [(UP, True), (UP, False)] * 2 + [(W, True), (W, False)]
