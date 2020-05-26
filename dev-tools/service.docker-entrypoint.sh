#!/bin/bash

set -euo pipefail

dockerize -wait "tcp://pgsql-dev:${PGSQL_DEV_PORT}"
exec "$@"
