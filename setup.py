"""Compatibility shim for legacy tooling.

All project metadata and dependencies are declared in ``pyproject.toml``
(PEP 621) — this file intentionally carries no configuration of its own so
that nothing is defined in two places. It exists only so that workflows
which still invoke ``setup.py`` directly (or older ``pip`` versions without
full PEP 621/660 support) continue to work; modern installs
(``pip install -e .``) read ``pyproject.toml`` alone.
"""

from setuptools import setup

setup()
