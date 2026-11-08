"""Marrow: an offline behavioural-authentication prototype.

The package deliberately uses only the Python standard library so that a fresh
copy runs anywhere Python 3.10+ is installed.  It never opens a network socket,
never starts a subprocess, and never needs a capture driver.
"""

__version__ = "1.0.0"
FORMAT_VERSION = "marrow-recording-v1"
