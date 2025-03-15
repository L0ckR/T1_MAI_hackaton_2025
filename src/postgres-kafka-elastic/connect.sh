!/usr/bin/bash
curl -X POST -H "Content-Type: application/json" --data @connectors/elasticsearch-sink.json.json http://localhost:8083/connectors 
curl -X POST -H "Content-Type: application/json" --data @connectors/postgres-connector.json.json http://localhost:8083/connectors 