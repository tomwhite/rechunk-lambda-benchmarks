#!/usr/bin/env bash
#
#   [DRY=1] [SHARD=1] [CONC=500] [MEM=1024] [FRAC=20] benchmarks/era5.sh era5-10GB|era5-100GB|era5-1TB|era5
#
# DRY=1 only shows the plan. Otherwise MEM is applied to the deployed copy function.
#
set -euo pipefail

NAME=$1
case $NAME in
  era5-10GB) T=2480 ;;
  era5-100GB) T=25854 ;;
  era5-1TB) T=258540 ;;
  era5) T=350640 ;;
  *) echo "unknown size: $NAME" >&2; exit 1 ;;
esac

S=s3://rechunk-lambda-benchmarks/$NAME
CONC=${CONC:-500}
MEM=${MEM:-1024}
FRAC=${FRAC:-20}
TAG=$NAME-m$MEM-f$FRAC-c$CONC
if [ "${SHARD:-0}" = 1 ]; then
  SHARD_FLAG=--shard-intermediates
  TAG=$TAG-sharded
else
  SHARD_FLAG=--no-shard-intermediates
fi

cd "$(dirname "$0")/.."

uv run rechunk-lambda-benchmarks plan \
  --shape "$T,721,1440" --source-chunks 31,721,1440 \
  --target-chunks "$T,10,10" $SHARD_FLAG --memory $MEM --min-mem-fraction $FRAC
[ "${DRY:-0}" = 1 ] && exit 0

# On exit, so a failed run (e.g. on data left by an earlier one) prints it too
trap 'echo; echo "To delete the data: aws s3 rm --recursive --quiet $S/"' EXIT

export AWS_PAGER=""
aws lambda update-function-configuration --function-name copy --memory-size $MEM >/dev/null
aws lambda wait function-updated --function-name copy

mkdir -p stats

time uv run rechunk-lambda-benchmarks hashed-integers "$S/source" \
  --shape "$T,721,1440" --chunks 31,721,1440 \
  --concurrency $CONC --stats-file "stats/$TAG-hashed-integers.jsonl"

time uv run rechunk-lambda-benchmarks rechunk "$S/source" "$S/target" \
  --target-chunks "$T,10,10" $SHARD_FLAG --memory $MEM --min-mem-fraction $FRAC \
  --concurrency $CONC --stats-file "stats/$TAG-rechunk.jsonl"

time uv run rechunk-lambda-benchmarks validate-hashed-integers "$S/target" \
  --concurrency $CONC --stats-file "stats/$TAG-validate.jsonl"

# REPORT lines reach CloudWatch Logs a few seconds after the run
sleep 30

for step in hashed-integers:hashed-integers rechunk:copy validate:validate-hashed-integers; do
  echo; echo "=== ${step%%:*} ==="
  uv run rechunk-lambda-benchmarks cloudwatch-report "stats/$TAG-${step%%:*}.jsonl" \
    --function-name "${step##*:}"
done

uv run rechunk-lambda-benchmarks timeline "stats/$TAG-rechunk.jsonl" --function-name copy
