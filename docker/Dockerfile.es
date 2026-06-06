# Elasticsearch 8.17 with IK Analyzer plugin for Chinese text tokenization
#
# Build:  docker compose build elasticsearch
# Start:  docker compose up -d elasticsearch
#
# IK Analyzer provides:
#   ik_max_word — fine-grained tokenization (indexing)
#   ik_smart     — coarse-grained tokenization (search)
#   Example: "认证方式" → ik_max_word: ["认证", "方式"] / ik_smart: ["认证方式"]

FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.0

RUN elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/8.17.0
