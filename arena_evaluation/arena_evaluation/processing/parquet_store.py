from __future__ import annotations

import pathlib
import json
import polars as pl

from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle


def _without_empty_structs(value: object) -> object:
    """Empty dicts infer as zero-field structs, which Parquet cannot represent."""
    if isinstance(value, dict):
        if not value:
            return None
        return {k: _without_empty_structs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_empty_structs(v) for v in value]
    return value


def _writable_rows(rows: list[dict]) -> list[dict]:
    """Metric rows with empty dicts replaced by None, so an episode with no doors writes."""
    return [{k: _without_empty_structs(v) for k, v in row.items()} for row in rows]


class ParquetStore:
    """Reads and writes metric DataFrames to Parquet format with embedded metadata."""
    METADATA_KEY = "arena_evaluation_metadata"
    
    @staticmethod
    def write(df: pl.DataFrame, dest: pathlib.Path, metadata: RunMetadata | None = None) -> None:
        """Write DataFrame to Parquet, embedding optional RunMetadata."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = df.to_arrow()
        
        if metadata is not None:
            meta_dict = table.schema.metadata or {}
            meta_dict[ParquetStore.METADATA_KEY.encode()] = json.dumps(
                metadata.model_dump(exclude_none=True)
            ).encode()
            table = table.replace_schema_metadata(meta_dict)
            
        import pyarrow.parquet as pq
        pq.write_table(table, dest)


    @staticmethod
    def write_rows(
        rows: list[dict],
        dest: pathlib.Path,
        *,
        schema_overrides: dict | None = None,
        metadata: RunMetadata | None = None,
    ) -> None:
        """Build a frame from metric rows and write it, dropping empty dicts on the way."""
        df = pl.DataFrame(_writable_rows(rows), schema_overrides=schema_overrides, infer_schema_length=None)
        ParquetStore.write(df, dest, metadata)

    @staticmethod
    def read(source: pathlib.Path) -> tuple[pl.DataFrame, dict | None]:
        """Read Parquet file and return (DataFrame, metadata_dict)."""
        if not source.exists():
            raise FileNotFoundError(f"Parquet file not found: {source}")
            
        import pyarrow.parquet as pq
        table = pq.read_table(source)
        
        metadata_dict = None
        if table.schema.metadata:
            meta_bytes = table.schema.metadata.get(ParquetStore.METADATA_KEY.encode())
            if meta_bytes:
                metadata_dict = json.loads(meta_bytes.decode())
                
        df = pl.from_arrow(table)
        return df, metadata_dict

    @staticmethod
    def combine(sources: list[pathlib.Path], dest: pathlib.Path) -> None:
        """Combine multiple metrics.parquet files into a single combined file."""
        if not sources:
            return
            
        dfs = []
        for src in sources:
            try:
                df, _ = ParquetStore.read(src)
                dfs.append(df)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to read {src} for combining: {e}")
                
        if not dfs:
            return
            
        try:
            combined = pl.concat(dfs, how="diagonal_relaxed")
        except Exception as e:
            raise SchemaViolationError(f"Failed to combine parquet files due to schema mismatch: {e}")
            
        ParquetStore.write(combined, dest)


class TopicParquetStore:
    """Reads and writes TopicBundle DataFrames to per-topic Parquet files."""
    @staticmethod
    def write(bundles: dict[str, TopicBundle], dest_dir: pathlib.Path, overwrite: bool = False) -> None:
        """Write TopicBundle DataFrames to Parquet files using zstd compression."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        import dataclasses
        
        # Determine global fields that are shared among all robots to save space
        global_fields = {"peds", "episode_record", "tf", "tf_static", "semantic_snapshot"}
        global_written = set()
        
        for robot_name, bundle in bundles.items():
            robot_dir = dest_dir / robot_name
            robot_dir.mkdir(parents=True, exist_ok=True)
            
            for field in dataclasses.fields(bundle):
                df = getattr(bundle, field.name)
                if df is not None:
                    # Write globals to the root dir
                    if field.name in global_fields:
                        if field.name in global_written:
                            continue
                        final_path = dest_dir / f"{field.name}.parquet"
                        global_written.add(field.name)
                    else:
                        final_path = robot_dir / f"{field.name}.parquet"
                    
                    if isinstance(df, pl.LazyFrame):
                        if not overwrite and final_path.exists():
                            continue
                        df = df.collect()
                    
                    if not df.is_empty():
                        temp_path = final_path.with_suffix(".parquet.tmp")
                        df.write_parquet(temp_path, compression="zstd")
                        temp_path.rename(final_path)

    @staticmethod
    def read(source_dir: pathlib.Path) -> dict[str, TopicBundle] | None:
        """
        Read Parquet files from source_dir to reconstruct a dict of TopicBundles.
        Returns None if no parquet files exist.
        """
        if not source_dir.exists() or not source_dir.is_dir():
            return None
            
        bundles = {}
        
        def load_parquet(path):
            if path.exists():
                try:
                    lf = pl.scan_parquet(path)
                    if "time_ns" in lf.collect_schema().names():
                        lf = lf.sort("time_ns")
                    return lf
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to read parquet {path}: {e}")
            return None

        # Load global topics from root
        global_bundle = TopicBundle()
        for p in source_dir.glob("*.parquet"):
            if p.stem == "peds_fallback":
                continue
            lf = load_parquet(p)
            if lf is not None:
                setattr(global_bundle, p.stem, lf)
        if global_bundle.peds is None:
            global_bundle.peds = load_parquet(source_dir / "peds_fallback.parquet")
                
        # Load robot specific topics
        robot_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

        robot_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
            
        for robot_dir in robot_dirs:
            robot_name = robot_dir.name
            rb = TopicBundle()
            
            # Copy global references
            rb.peds = global_bundle.peds
            rb.episode_record = global_bundle.episode_record
            rb.tf = global_bundle.tf
            rb.tf_static = global_bundle.tf_static
            rb.semantic_snapshot = global_bundle.semantic_snapshot

            for p in robot_dir.glob("*.parquet"):
                lf = load_parquet(p)
                if lf is not None:
                    setattr(rb, p.stem, lf)
                    
            bundles[robot_name] = rb
            
        return bundles if bundles else None

