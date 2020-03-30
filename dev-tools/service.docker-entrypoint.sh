#!/bin/bash
#echo 'hello world'
set -euo pipefail

#DB=($(from_vault income-service database_host database_name database_username database_password))

export POSTGRES_HOST=pgsql-dev
export POSTGRES_DB=database
export POSTGRES_USER=service
export POSTGRES_PASSWORD=password
export POSTGRES_PORT=${PGSQL_DEV_PORT}