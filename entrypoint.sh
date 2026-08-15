#!/bin/sh
set -e

ARGS=""

if [ -n "$APP_USERNAME" ] && [ -n "$APP_PASSWORD" ]; then
    ARGS="--username $APP_USERNAME --password $APP_PASSWORD"
fi

# shellcheck disable=SC2086
exec python3 app.py --server-name 0.0.0.0 --port 6969 $ARGS