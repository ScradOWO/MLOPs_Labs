"""Answer lineage questions from the ML Metadata store written by pipeline.py.

    python lineage.py runs                   # every pipeline run, its steps and their states
    python lineage.py trace [MODEL_ID]       # what produced this model? (default: newest model)
    python lineage.py impact DATA_VERSION    # what was built from this dataset? (e.g. a bad-data recall)
"""
import argparse
import os
import sys

from store import LAB_DIR, connect, mlmd

STATE_NAMES = {v: k for k, v in mlmd.Execution.State.items()}


def describe_artifact(a):
    props = " ".join(f"{k}={getattr(v, v.WhichOneof('value'))}"
                     for k, v in sorted(a.properties.items()) if k != "fingerprint")
    return f"{a.type} #{a.id} {props}  [{os.path.relpath(a.uri, LAB_DIR)}]"


def describe_execution(store, ex):
    runs = ",".join(c.name for c in store.get_contexts_by_execution(ex.id))
    return f"{ex.type} #{ex.id} {STATE_NAMES[ex.last_known_state]} in {runs}"


def walk(store, artifact_id, upstream, depth=0):
    """Print the lineage tree of one artifact by following events.

    upstream:   artifact <- OUTPUT of execution <- its INPUT artifacts <- ...
    downstream: artifact -> INPUT of execution -> its OUTPUT artifacts -> ...
    """
    [artifact] = store.get_artifacts_by_id([artifact_id])
    print("    " * depth + describe_artifact(artifact))
    link, follow = (mlmd.Event.OUTPUT, mlmd.Event.INPUT) if upstream else (mlmd.Event.INPUT, mlmd.Event.OUTPUT)
    arrow = "<-" if upstream else "->"
    ex_ids = [e.execution_id for e in store.get_events_by_artifact_ids([artifact_id]) if e.type == link]
    for ex in store.get_executions_by_id(ex_ids):
        print("    " * depth + f"  {arrow} {describe_execution(store, ex)}")
        for e in store.get_events_by_execution_ids([ex.id]):
            if e.type == follow:
                walk(store, e.artifact_id, upstream, depth + 1)


def cmd_runs(store, _):
    for ctx in sorted(store.get_contexts_by_type("PipelineRun"), key=lambda c: c.id):
        steps = sorted(store.get_executions_by_context(ctx.id), key=lambda e: e.id)
        summary = ", ".join(f"{ex.type}:{STATE_NAMES[ex.last_known_state]}" for ex in steps)
        models = [a for a in store.get_artifacts_by_context(ctx.id) if a.type == "Model"]
        result = " ".join(f"Model #{m.id} acc={m.properties['accuracy'].double_value}" for m in models) or "no model"
        print(f"{ctx.name}  data={ctx.properties['data_version'].string_value:<3} "
              f"n_estimators={ctx.properties['n_estimators'].int_value:<4} {summary:<55} -> {result}")


def cmd_trace(store, args):
    models = store.get_artifacts_by_type("Model")
    if not models:
        sys.exit("No models in the store yet. Run pipeline.py first.")
    model_id = args.model_id or max(m.id for m in models)
    walk(store, model_id, upstream=True)


def cmd_impact(store, args):
    datasets = [a for a in store.get_artifacts_by_type("DataSet")
                if a.properties["version"].string_value == args.data_version]
    if not datasets:
        sys.exit(f"No DataSet with version {args.data_version} in the store.")
    for d in datasets:
        walk(store, d.id, upstream=False)
    ex_ids = {e.execution_id for e in store.get_events_by_artifact_ids([d.id for d in datasets])
              if e.type == mlmd.Event.INPUT}
    out_ids = {e.artifact_id for e in store.get_events_by_execution_ids(list(ex_ids))
               if e.type == mlmd.Event.OUTPUT}
    models = [a for a in store.get_artifacts_by_id(sorted(out_ids)) if a.type == "Model"]
    print(f"\nModels built from {args.data_version}: {', '.join(f'#{m.id}' for m in models) or 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs")
    sub.add_parser("trace").add_argument("model_id", type=int, nargs="?")
    sub.add_parser("impact").add_argument("data_version")
    args = parser.parse_args()
    {"runs": cmd_runs, "trace": cmd_trace, "impact": cmd_impact}[args.cmd](connect(), args)


if __name__ == "__main__":
    main()
