"""Build version. The deploy workflow overwrites this file with the release version.

It is tracked on purpose: ``hf upload`` skips files that are in .gitignore.
"""

__version__ = "dev"
__commit__ = None
