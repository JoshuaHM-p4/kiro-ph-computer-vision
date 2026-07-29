"""Shared building blocks for the demo suite.

Import boundaries matter here:

* ``geometry``, ``landmarks`` and ``gestures`` are pure. They never touch a
  camera or open a window, so demo logic built on them is unit testable.
* ``detectors`` and ``camera`` are the desktop-only vision layer.
* ``hud`` draws the shared neon theme onto numpy frames.
* ``webapp`` is the Flask/WebSocket adapter for the browser path.
"""
