#!/bin/bash

set -euo pipefail

DB=($(from_vault income-service database_host database_name database_username database_password))

export POSTGRES_HOST=${DB[0]}
export POSTGRES_DB=${DB[1]}
export POSTGRES_USER=${DB[2]}
export POSTGRES_PASSWORD=${DB[3]}
export POSTGRES_PORT=${PGSQL_DEV_PORT}