"""Run one pass of a small training pipeline and record every step in ML Metadata.

    python pipeline.py --data-version v1 [--n-estimators 100]

Steps: SchemaGen (first run only) -> DataValidation -> Trainer.
Each pass is one PipelineRun context; every step is an Execution linked to its
input and output artifacts by events. The Trainer asks the store whether it
already trained on the same inputs and parameters, and reuses that model if so.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from store import LAB_DIR, connect, mlmd, register_types

ART_DIR = Path(os.environ.get("ARTIFACT_DIR", LAB_DIR / "artifacts"))
STATE = mlmd.Execution.State


def make_dataset(version):
    """Simulate three data drops: v1 = first 60% of rows, v2 = all rows, v3 = all rows but corrupted."""
    df = load_breast_cancer(as_frame=True).frame
    if version == "v1":
        df = df.sample(frac=0.6, random_state=0)
    elif version == "v3":
        df = df.copy()
        df.loc[df.index[:5], "mean radius"] = -1.0  # impossible negative measurement
        df.loc[df.index[5:10], "mean texture"] = np.nan  # missing values
    path = ART_DIR / "data" / f"{version}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path, df


def new_artifact(ids, type_name, uri, **props):
    a = mlmd.Artifact(type_id=ids[type_name], uri=str(uri), state=mlmd.Artifact.LIVE)
    for key, value in props.items():
        set_value(a.properties[key], value)
    return a


def set_value(field, value):
    if isinstance(value, int):
        field.int_value = value
    elif isinstance(value, float):
        field.double_value = value
    else:
        field.string_value = value


def record(store, ids, ctx, exec_type, state, inputs=(), outputs=(), **props):
    """Write one execution, its input/output events and its context links in a single transaction."""
    ex = mlmd.Execution(type_id=ids[exec_type], last_known_state=state)
    for key, value in props.items():
        set_value(ex.properties[key], value)
    events = [(a, mlmd.Event(type=mlmd.Event.INPUT)) for a in inputs]
    events += [(a, mlmd.Event(type=mlmd.Event.OUTPUT)) for a in outputs]
    ex_id, artifact_ids, _ = store.put_execution(ex, events, [ctx])
    for a, a_id in zip([*inputs, *outputs], artifact_ids):
        a.id = a_id  # new artifacts get their ids here
    return ex_id


def get_or_create_dataset(store, ids, version, path, df):
    """Same file content -> same DataSet artifact, so lineage and caching see one dataset, not one per run."""
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    # filter_query was removed in ml-metadata 1.21, so query by type and filter here.
    for a in store.get_artifacts_by_type("DataSet"):
        if a.properties["fingerprint"].string_value == fingerprint:
            return a
    return new_artifact(ids, "DataSet", path.resolve(), version=version, rows=len(df), fingerprint=fingerprint)


def infer_schema(df):
    return {
        col: {"dtype": str(df[col].dtype), "nullable": bool(df[col].isna().any()), "non_negative": bool(df[col].min() >= 0)}
        for col in df.columns
    }


def validate(df, schema):
    anomalies = []
    for col, rule in schema.items():
        if col not in df.columns:
            anomalies.append(f"{col}: column missing")
            continue
        if str(df[col].dtype) != rule["dtype"]:
            anomalies.append(f"{col}: dtype {df[col].dtype}, expected {rule['dtype']}")
        if not rule["nullable"] and df[col].isna().any():
            anomalies.append(f"{col}: {int(df[col].isna().sum())} missing values")
        if rule["non_negative"] and (df[col] < 0).any():
            anomalies.append(f"{col}: {int((df[col] < 0).sum())} negative values")
    return anomalies


def find_cached_model(store, n_estimators, input_ids):
    """Return the Model of an earlier COMPLETE Trainer run with the same inputs and params, if its file still exists."""
    for ex in store.get_executions_by_type("Trainer"):
        if ex.last_known_state != STATE.COMPLETE or ex.properties["n_estimators"].int_value != n_estimators:
            continue
        events = store.get_events_by_execution_ids([ex.id])
        if {e.artifact_id for e in events if e.type == mlmd.Event.INPUT} != input_ids:
            continue
        [model_id] = [e.artifact_id for e in events if e.type == mlmd.Event.OUTPUT]
        [model] = store.get_artifacts_by_id([model_id])
        if Path(model.uri).exists():
            return model
    return None


def train(df, n_estimators, out_path):
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(X_train, y_train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return round(float(model.score(X_test, y_test)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-version", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    store = connect()
    ids = register_types(store)

    run_name = f"run-{len(store.get_contexts_by_type('PipelineRun')) + 1:03d}"
    ctx = mlmd.Context(type_id=ids["PipelineRun"], name=run_name)
    ctx.properties["data_version"].string_value = args.data_version
    ctx.properties["n_estimators"].int_value = args.n_estimators
    [ctx.id] = store.put_contexts([ctx])
    print(f"== {run_name}: data {args.data_version}, n_estimators={args.n_estimators}")

    path, df = make_dataset(args.data_version)
    dataset = get_or_create_dataset(store, ids, args.data_version, path, df)

    # The schema is learned once, from the first dataset, and every later run validates against it.
    schemas = store.get_artifacts_by_type("Schema")
    if schemas:
        schema_artifact = max(schemas, key=lambda a: a.id)
        print(f"Schema         reusing Schema #{schema_artifact.id} from the store")
    else:
        schema_path = ART_DIR / "schema" / "schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(infer_schema(df), indent=2))
        schema_artifact = new_artifact(ids, "Schema", schema_path.resolve(), columns=len(df.columns))
        record(store, ids, ctx, "SchemaGen", STATE.COMPLETE, [dataset], [schema_artifact])
        print(f"SchemaGen      inferred Schema #{schema_artifact.id} from {args.data_version}")
    if not Path(schema_artifact.uri).exists():
        sys.exit(f"{schema_artifact.uri} is recorded in the store but missing on disk. "
                 "Delete metadata/ and artifacts/ together to start over.")
    schema = json.loads(Path(schema_artifact.uri).read_text())

    anomalies = validate(df, schema)
    report_path = ART_DIR / "validation" / f"{run_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(anomalies, indent=2))
    report = new_artifact(ids, "ValidationReport", report_path.resolve(), anomalies=len(anomalies))
    state = STATE.FAILED if anomalies else STATE.COMPLETE
    record(store, ids, ctx, "DataValidation", state, [dataset, schema_artifact], [report])
    if anomalies:
        print(f"DataValidation FAILED with {len(anomalies)} anomalies:")
        for line in anomalies:
            print(f"                 - {line}")
        sys.exit(f"Pipeline stopped: {run_name} recorded as FAILED, no model trained.")
    print(f"DataValidation passed (DataSet #{dataset.id}, {dataset.properties['rows'].int_value} rows)")

    inputs = [dataset, schema_artifact]
    model = find_cached_model(store, args.n_estimators, {a.id for a in inputs})
    if model:
        record(store, ids, ctx, "Trainer", STATE.CACHED, inputs, [model], n_estimators=args.n_estimators)
        print(f"Trainer        CACHED: same inputs and params already trained, reusing Model #{model.id}")
    else:
        model_path = ART_DIR / "models" / f"{run_name}.joblib"
        try:
            accuracy = train(df, args.n_estimators, model_path)
        except Exception:
            record(store, ids, ctx, "Trainer", STATE.FAILED, inputs, n_estimators=args.n_estimators)
            raise
        model = new_artifact(ids, "Model", model_path.resolve(), n_estimators=args.n_estimators, accuracy=accuracy)
        record(store, ids, ctx, "Trainer", STATE.COMPLETE, inputs, [model], n_estimators=args.n_estimators)
        print(f"Trainer        trained Model #{model.id}")
    print(f"Result         Model #{model.id} test accuracy {model.properties['accuracy'].double_value}")


if __name__ == "__main__":
    main()
