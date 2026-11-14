"""Connect to the persistent ML Metadata store and register the lab's types.

Shared by pipeline.py (writes metadata) and lineage.py (reads it).
"""
import os
from pathlib import Path

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2 as mlmd

LAB_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("MLMD_DB", LAB_DIR / "metadata" / "mlmd.sqlite"))

STRING, INT, DOUBLE = mlmd.STRING, mlmd.INT, mlmd.DOUBLE

# name -> properties. put_*_type() is idempotent, so registering on every run is safe.
ARTIFACT_TYPES = {
    "DataSet": {"version": STRING, "rows": INT, "fingerprint": STRING},
    "Schema": {"columns": INT},
    "ValidationReport": {"anomalies": INT},
    "Model": {"n_estimators": INT, "accuracy": DOUBLE},
}
EXECUTION_TYPES = {
    "SchemaGen": {},
    "DataValidation": {},
    "Trainer": {"n_estimators": INT},
}
CONTEXT_TYPES = {
    "PipelineRun": {"data_version": STRING, "n_estimators": INT},
}


def connect(db_path=DB_PATH):
    """Open (or create) the SQLite-backed store. Unlike the original lab's fake database, it survives between runs."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = mlmd.ConnectionConfig()
    config.sqlite.filename_uri = str(db_path)
    config.sqlite.connection_mode = mlmd.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
    return metadata_store.MetadataStore(config)


def register_types(store):
    """Register every type and return {type name: type id}."""
    ids = {}
    for proto, put, types in [
        (mlmd.ArtifactType, store.put_artifact_type, ARTIFACT_TYPES),
        (mlmd.ExecutionType, store.put_execution_type, EXECUTION_TYPES),
        (mlmd.ContextType, store.put_context_type, CONTEXT_TYPES),
    ]:
        for name, props in types.items():
            t = proto(name=name)
            for key, kind in props.items():
                t.properties[key] = kind
            ids[name] = put(t)
    return ids
