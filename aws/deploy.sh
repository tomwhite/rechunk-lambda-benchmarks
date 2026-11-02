#!/usr/bin/env bash
set -euo pipefail

export AWS_PAGER=""

cd "$(dirname "$0")/../rust"

cargo lambda build --release --arm64

worker_role=$(aws iam get-role --role-name rechunk-lambda-benchmarks-worker-role --query Role.Arn --output text)
dispatcher_role=$(aws iam get-role --role-name map-lambda-dispatcher-role --query Role.Arn --output text)

for f in copy hashed-integers validate-hashed-integers delete-intermediate; do
  cargo lambda deploy --binary-name $f --role "$worker_role" $f
done
cargo lambda deploy --binary-name dispatcher --role "$dispatcher_role" dispatcher
