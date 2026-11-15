#!/usr/bin/env bash
# Runs once in the `setup` service: index template in Elasticsearch, data views + dashboard in Kibana.
set -euo pipefail
ES=http://elasticsearch:9200
KB=http://kibana:5601

echo "Creating index template ml-logs"
curl -fsS -X PUT "$ES/_index_template/ml-logs" -H 'Content-Type: application/json' \
  -d @/setup/index-template.json
echo

echo "Importing data views and dashboard into Kibana"
result=$(curl -fsS -X POST "$KB/api/saved_objects/_import?overwrite=true" -H 'kbn-xsrf: true' \
  --form file=@/setup/dashboard.ndjson)
echo "$result"
grep -q '"success":true' <<<"$result" || { echo "Kibana import failed"; exit 1; }
echo "SETUP DONE"
