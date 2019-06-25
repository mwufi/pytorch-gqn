#!/usr/bin/env bash

# Example usage:
# ./download-data.sh data

LOCATION=$1

echo "Downloading data"
gsutil -m cp -R gs://gqn-dataset/shepard_metzler_5_parts $LOCATION