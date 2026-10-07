"""Observe selected module imports once, without importing optional packages."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


class _Loader(importlib.abc.Loader):
    def __init__(self, original, finder, fullname):
        self.original = original
        self.finder = finder
        self.fullname = fullname

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            self.finder.activate(module)
        finally:
            self.finder.consume(self.fullname)


class _ImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, targets, activate):
        self.targets = set(targets)
        self.activate = activate

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self, fullname)
        return spec

    def consume(self, fullname):
        self.targets.discard(fullname)
        if not self.targets:
            uninstall(self)


def install(targets, activate, *, ready_attribute="Trainer"):
    pending = []
    for name in targets:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, ready_attribute):
            activate(module)
        else:
            pending.append(name)
    if not pending:
        return None
    finder = _ImportFinder(pending, activate)
    sys.meta_path.insert(0, finder)
    return finder


def uninstall(finder):
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
