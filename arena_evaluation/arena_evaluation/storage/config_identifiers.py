"""Config bundles as resolvable identifiers: one chain for every declaration kind.

A suite, contest, report manifest, plot or query resolves in this order: the imported assets root,
the package configs (installed share, then the source tree), the local cache of each benchmark
bucket (BENCHMARK_BUCKETS), and the bucket itself. A local copy always shadows a bucket copy, so
publishing a declaration never overrides a local one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from arena_simulation_setup import ARENA_ASSETS_DIR, NO_ASSETS_DIR
from arena_simulation_setup.tree import (
    BENCHMARK_PROVIDERS,
    AssetIdentifier,
    DynamicPaths,
    FallbackResolver,
    NetResolver,
    SimplePathResolver,
)

from arena_evaluation.storage.config_layout import (
    CONFIG_KINDS,
    CONFIGS_DIR_NAME,
    all_config_roots,
    share_dir,
)

__all__ = [
    "ConfigPathResolver",
    "bucket_cache_dirs",
    "bundle_filename",
    "config_dirs",
    "register_config_identifier",
]

_BENCH_TTL_S = 365 * 86400


def bundle_filename(asset_type: str) -> str:
    """The yaml a directory bundle holds, found by its folder: ``suites`` -> ``suite.yaml``, ``queries`` -> ``query.yaml``."""
    return next(
        spec.filename for spec in CONFIG_KINDS.values() if spec.folder == asset_type
    )


class ConfigPathResolver(SimplePathResolver):
    """Resolves directory bundles ``<folder>/<name>/<kind>.yaml`` only."""

    async def resolve(self, identifier: AssetIdentifier) -> Path | None:
        if identifier in self._cache:
            return self._cache[identifier]
        bundled = self.path / identifier.relpath() / bundle_filename(self._asset_type)
        if bundled.is_file():
            self._cache[identifier] = bundled
            return bundled
        return None

    def listall(self, **kwargs: object) -> Iterator[AssetIdentifier]:
        del kwargs
        filename = bundle_filename(self._asset_type)
        source = self.path / self._asset_type
        if not source.is_dir():
            return
        for entry in sorted(source.iterdir()):
            if entry.is_dir() and (entry / filename).is_file():
                yield self._IdentifierT(name=entry.name)


def config_dirs() -> Iterator[Path]:
    """The imported assets root, then the package configs (installed share, then source tree)."""
    yield from all_config_roots(DynamicPaths.IMPORTED.path)


def bucket_cache_dirs() -> list[Path]:
    """The local cache root of each benchmark bucket that has been fetched into before."""
    if ARENA_ASSETS_DIR == NO_ASSETS_DIR:
        return []
    return [
        ARENA_ASSETS_DIR / provider
        for provider in BENCHMARK_PROVIDERS
        if (ARENA_ASSETS_DIR / provider).is_dir()
    ]


def register_config_identifier(cls: type[AssetIdentifier]) -> None:
    """Chain a config identifier: local config folders, then the benchmark buckets, then a write target."""
    cls.use(*(ConfigPathResolver(cls, d) for d in config_dirs()))
    cls.use(
        *NetResolver.all(
            cls,
            providers=BENCHMARK_PROVIDERS,
            formats=(),
            ttl=_BENCH_TTL_S,
            annotated=False,
            list_prefix=cls._asset_type,
        )
    )
    share = share_dir()
    if share is not None:
        cls.use(FallbackResolver(cls, share / CONFIGS_DIR_NAME))
