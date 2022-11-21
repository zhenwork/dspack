#!/bin/bash

DSPACK_SCRIPT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

## ADD DSPACK_HOME
export DSPACK_HOME=${DSPACK_SCRIPT_HOME}
echo "export DSPACK_HOME=${DSPACK_SCRIPT_HOME}"

## ADD DSPACK_HOME_APP
if [[ $PATH == *"${DSPACK_SCRIPT_HOME}/app:"* ]]; then
    echo "\$PATH contains ${DSPACK_SCRIPT_HOME}/app:"
elif [[ $PATH == *"${DSPACK_SCRIPT_HOME}/app/:"* ]]; then
    echo "\$PATH contains ${DSPACK_SCRIPT_HOME}/app/:"
else
    export PATH=${DSPACK_SCRIPT_HOME}/app:$PATH
    echo "export PATH=${DSPACK_SCRIPT_HOME}/app:\$PATH"
fi

## Make all sh files exectuable
cd ${DSPACK_SCRIPT_HOME}/app && chmod +x * && cd -
