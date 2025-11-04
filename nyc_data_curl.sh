#!/usr/bin/env bash
set -euo pipefail

# URLs to fetch (base URLs for month 01; the script will replace the month number for 01-12)
URLS=(
  "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
  "https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2023-01.parquet"
  "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet"
)

# create 01-12 directories
for m in $(seq -w 1 12); do
    mkdir -p "./data/2023/$m"
done

# iterate the explicit URLs and curl per month, replacing the month digits
for url in "${URLS[@]}"; do
    for m in $(seq -w 1 12); do
        newurl=$(sed -E "s/([0-9]{4}-)[0-9]{2}(\.parquet)\$/\1$m\2/" <<<"$url")
        out="./data/2023/$m/$(basename "$newurl")"
        curl -fSL "$newurl" -o "$out" || echo "curl failed: $newurl" >&2
    done
done