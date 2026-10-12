#!/usr/bin/env bash

export AWS_PAGER=""

WORKER_ROLE=rechunk-lambda-benchmarks-worker-role
COORDINATOR_ROLE=map-lambda-coordinator-role

for f in copy hashed-integers validate-hashed-integers delete-intermediate coordinator; do
  aws logs delete-log-group --log-group-name /aws/lambda/$f
  aws lambda delete-function --function-name $f
done

aws iam delete-role-policy --role-name $WORKER_ROLE --policy-name rechunk-lambda-benchmarks-s3
aws iam delete-role-policy --role-name $COORDINATOR_ROLE --policy-name invoke-rechunk-lambda-benchmarks-workers

for role in $WORKER_ROLE $COORDINATOR_ROLE; do
  aws iam detach-role-policy --role-name $role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam delete-role --role-name $role
done
