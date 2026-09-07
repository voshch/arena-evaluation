"""Suites, contests, and report manifests as Identifiers, resolved through the
same chain (local share dir -> source tree -> network -> local fallback) as
every other asset in :mod:`arena_simulation_setup.tree`."""

from __future__ import annotations

import typing
from collections.abc import Iterator
from pathlib import Path

import yaml
from arena_simulation_setup.tree import (
    BENCHMARK_PROVIDERS,
    AssetIdentifier,
    FallbackResolver,
    NetResolver,
    SimplePathResolver,
)

from arena_evaluation.presentation.manifest_registry import share_dir, source_tree_dir

if typing.TYPE_CHECKING:
    from arena_evaluation.presentation.viz_manifest import VizManifest
    from .config import Contest, Suite

_BENCH_TTL_S = 365 * 86400


def _stem(asset_type: str) -> str:
    """`suites` -> `suite`: the filename a directory bundle stores its config under."""
    return asset_type.removesuffix('s')


def _config_file(path: Path, asset_type: str) -> Path:
    """The yaml to read. A local resolver hands back the file itself, a bucket hands back
    the bundle directory holding it, and load() has to accept either."""
    return path / f'{_stem(asset_type)}.yaml' if path.is_dir() else path


class SuiteIdentifier(AssetIdentifier["Suite"]):
    _asset_type = 'suites'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SuiteIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> Suite:
        del kwargs
        from .config import Suite

        return Suite.parse(self.name, yaml.safe_load(_config_file(path, self._asset_type).read_text()))


class ContestIdentifier(AssetIdentifier["Contest"]):
    _asset_type = 'contests'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContestIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> Contest:
        del kwargs
        from .config import Contest

        return Contest.parse(self.name, yaml.safe_load(_config_file(path, self._asset_type).read_text()))


class ManifestIdentifier(AssetIdentifier['VizManifest']):
    _asset_type = 'manifests'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ManifestIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> VizManifest:
        del kwargs
        from arena_evaluation.presentation.viz_manifest import VizManifest

        return VizManifest.load(_config_file(path, self._asset_type))


class ConfigPathResolver(SimplePathResolver):
    """Flat `<name>.yaml` wins over the directory-bundle `<name>/<kind>.yaml`."""

    async def resolve(self, identifier: AssetIdentifier) -> Path | None:
        if identifier in self._cache:
            return self._cache[identifier]
        kind = _stem(self._asset_type)
        stem = self.path / identifier.relpath()
        flat = stem.with_suffix('.yaml')
        if flat.is_file():
            self._cache[identifier] = flat
            return flat
        bundled = stem / f'{kind}.yaml'
        if bundled.is_file():
            self._cache[identifier] = bundled
            return bundled
        return None

    def listall(self, **kwargs: object) -> Iterator[AssetIdentifier]:
        del kwargs
        kind = _stem(self._asset_type)
        source = self.path / self._asset_type
        if not source.is_dir():
            return
        for entry in sorted(source.iterdir()):
            if entry.is_file() and entry.suffix == '.yaml':
                yield self._IdentifierT(name=entry.stem)
            elif entry.is_dir() and (entry / f'{kind}.yaml').is_file():
                yield self._IdentifierT(name=entry.name)

    async def listall_async(self, **kwargs: object) -> list[AssetIdentifier]:
        # the inherited child-process walk would list raw filenames, not config names
        return list(self.listall(**kwargs))


def _bench_dirs() -> Iterator[Path]:
    """configs/benchmark under the package share dir, then the source-tree fallback."""
    for base in (share_dir(), source_tree_dir()):
        if base is not None:
            yield base / 'configs' / 'benchmark'


def _register(cls: type[AssetIdentifier]) -> None:
    cls.use(*(ConfigPathResolver(cls, d) for d in _bench_dirs()))
    cls.use(*NetResolver.all(cls, providers=BENCHMARK_PROVIDERS, formats=(), ttl=_BENCH_TTL_S, annotated=False, list_prefix=cls._asset_type))
    share = share_dir()
    if share is not None:
        cls.use(FallbackResolver(cls, share / 'configs' / 'benchmark'))


_register(SuiteIdentifier)
_register(ContestIdentifier)
_register(ManifestIdentifier)
