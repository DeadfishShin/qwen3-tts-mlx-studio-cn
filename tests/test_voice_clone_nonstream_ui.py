import inspect

import ui.tabs.voice_clone as voice_clone_tab
from ui import components


def test_single_clone_hides_non_interruptible_stop_control():
    source = inspect.getsource(voice_clone_tab.wire)
    assert "show_stop=False" in source


def test_other_lifecycles_keep_stop_visible_by_default():
    source = inspect.getsource(components.wire_run_lifecycle)
    assert "show_stop=True" in source
