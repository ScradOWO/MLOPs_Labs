"""Generate kibana/dashboard.ndjson: two data views, three Vega-Lite panels and the dashboard.

    python kibana/build_dashboard.py

Saved objects nest JSON inside JSON strings, so they are built here instead of
hand-edited. The output file is committed; re-run this only after changing a panel.
"""
import json
from pathlib import Path

FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
EMPTY_SEARCH = json.dumps({"query": {"query": "", "language": "kuery"}, "filter": []})


def histogram(index, extra_aggs):
    """Vega data source: 10 s buckets of `index`, following the dashboard's time picker."""
    return {
        "url": {
            "index": index,
            "%context%": True,
            "%timefield%": "@timestamp",
            "body": {
                "size": 0,
                "aggs": {"t": {"date_histogram": {"field": "@timestamp", "fixed_interval": "10s"},
                               "aggs": extra_aggs}},
            },
        },
        "format": {"property": "aggregations.t.buckets"},
    }


PANELS = {
    "ml-traffic": ("Requests and mean latency per 10 s", {
        "$schema": SCHEMA,
        "data": histogram("ml-predictions", {"latency": {"avg": {"field": "latency_ms"}}}),
        "encoding": {"x": {"field": "key", "type": "temporal", "title": None}},
        "layer": [
            {"mark": {"type": "bar", "opacity": 0.6, "size": 22},
             "encoding": {"y": {"field": "doc_count", "type": "quantitative", "title": "requests"}}},
            {"mark": {"type": "line", "color": "#e7664c", "point": True},
             "encoding": {"y": {"field": "latency.value", "type": "quantitative", "title": "mean latency (ms)"}}},
        ],
        "resolve": {"scale": {"y": "independent"}},
    }),
    "ml-prediction-mean": ("Mean prediction per 10 s (output drift)", {
        "$schema": SCHEMA,
        "data": histogram("ml-predictions", {"pred": {"avg": {"field": "prediction"}}}),
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "key", "type": "temporal", "title": None},
            "y": {"field": "pred.value", "type": "quantitative", "title": "mean predicted progression",
                  "scale": {"zero": False}},
        },
    }),
    "ml-psi": ("PSI per feature (>= 0.2 = drift alert)", {
        "$schema": SCHEMA,
        "data": {
            "url": {"index": "ml-drift", "%context%": True, "%timefield%": "@timestamp",
                    "body": {"size": 1000, "_source": ["@timestamp", "psi", "prediction_psi"]}},
            "format": {"property": "hits.hits"},
        },
        "transform": [
            {"calculate": "toDate(datum._source['@timestamp'])", "as": "time"},
            {"fold": [f"_source.psi.{f}" for f in FEATURES] + ["_source.prediction_psi"]},
            {"calculate": "replace(replace(datum.key, '_source.psi.', ''), '_source.prediction_psi', 'prediction')",
             "as": "feature"},
        ],
        "layer": [
            {"mark": {"type": "line", "point": True},
             "encoding": {
                 "x": {"field": "time", "type": "temporal", "title": None},
                 "y": {"field": "value", "type": "quantitative", "title": "PSI"},
                 "color": {"field": "feature", "type": "nominal", "scale": {"scheme": "tableau20"}},
                 "tooltip": [{"field": "feature"}, {"field": "value", "format": ".3f"},
                             {"field": "time", "type": "temporal", "timeUnit": "hoursminutesseconds"}],
             }},
            {"mark": {"type": "rule", "color": "red", "strokeDash": [6, 4]},
             "encoding": {"y": {"datum": 0.2}}},
        ],
    }),
}


def objects():
    for index, name in [("ml-predictions", "ML predictions"), ("ml-drift", "ML drift checks")]:
        yield {"type": "index-pattern", "id": f"dv-{index}",
               "attributes": {"title": index, "name": name, "timeFieldName": "@timestamp"}, "references": []}

    for vis_id, (title, spec) in PANELS.items():
        yield {"type": "visualization", "id": vis_id, "references": [], "attributes": {
            "title": title,
            "description": "",
            "uiStateJSON": "{}",
            "visState": json.dumps({"title": title, "type": "vega", "aggs": [],
                                    "params": {"spec": json.dumps(spec, indent=2)}}),
            "kibanaSavedObjectMeta": {"searchSourceJSON": EMPTY_SEARCH},
        }}

    layout = [(0, 0, 24, 14), (24, 0, 24, 14), (0, 14, 48, 18)]  # x, y, w, h on Kibana's 48-column grid
    panels, refs = [], []
    for i, (vis_id, (x, y, w, h)) in enumerate(zip(PANELS, layout)):
        panels.append({"panelIndex": str(i), "gridData": {"x": x, "y": y, "w": w, "h": h, "i": str(i)},
                       "type": "visualization", "panelRefName": f"panel_{i}", "embeddableConfig": {}})
        refs.append({"name": f"panel_{i}", "type": "visualization", "id": vis_id})
    yield {"type": "dashboard", "id": "ml-monitoring", "references": refs, "attributes": {
        "title": "ML model monitoring",
        "description": "Traffic, latency, prediction drift and per-feature PSI for the diabetes model",
        "panelsJSON": json.dumps(panels),
        "optionsJSON": json.dumps({"useMargins": True, "hidePanelTitles": False}),
        "timeRestore": True,
        "timeFrom": "now-15m",
        "timeTo": "now",
        "refreshInterval": {"pause": False, "value": 10000},
        "kibanaSavedObjectMeta": {"searchSourceJSON": EMPTY_SEARCH},
    }}


if __name__ == "__main__":
    out = Path(__file__).with_name("dashboard.ndjson")
    out.write_text("".join(json.dumps(o) + "\n" for o in objects()))
    print(f"wrote {out}")
