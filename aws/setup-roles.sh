#!/usr/bin/env bash

export AWS_PAGER=""

BUCKET=rechunk-lambda-benchmarks
WORKER_ROLE=rechunk-lambda-benchmarks-worker-role
COORDINATOR_ROLE=map-lambda-coordinator-role

# Roles Lambda can assume, with permission to write logs.
# The coordinator role is shared with other map-lambda projects, so it may already exist.
for role in $WORKER_ROLE $COORDINATOR_ROLE; do
  aws iam create-role --role-name $role --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' >/dev/null
  aws iam attach-role-policy --role-name $role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
done

# Inline policy names must be unique across projects sharing the coordinator role
aws iam put-role-policy --role-name $COORDINATOR_ROLE \
  --policy-name invoke-rechunk-lambda-benchmarks-workers --policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": [
      "arn:aws:lambda:*:*:function:copy",
      "arn:aws:lambda:*:*:function:hashed-integers",
      "arn:aws:lambda:*:*:function:validate-hashed-integers"
    ]
  }]
}'

# ListBucket makes a GET for a missing key 404 rather than 403.
# Delete only intermediates, matching the guard in delete-intermediate.
aws iam put-role-policy --role-name $WORKER_ROLE \
  --policy-name rechunk-lambda-benchmarks-s3 --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::'$BUCKET'/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::'$BUCKET'"
    },
    {
      "Effect": "Allow",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::'$BUCKET'/*/_intermediate/*"
    }
  ]
}'
