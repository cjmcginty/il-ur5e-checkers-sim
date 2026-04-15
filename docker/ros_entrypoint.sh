#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

WS=/workspaces/ur5e-checkers-irl

if [ -d "$WS/src" ]; then
  cd "$WS"

  # Build if install/setup.bash is missing, or if any source file is newer than install/
  if [ ! -f "$WS/install/setup.bash" ] || [ -n "$(find src -type f -newer install/setup.bash 2>/dev/null | head -n 1)" ]; then
    echo "[entrypoint] Building workspace..."
    colcon build --symlink-install
  fi

  if [ -f "$WS/install/setup.bash" ]; then
    source "$WS/install/setup.bash"
  fi
fi

exec "$@"