# il-ur5e-checkers-sim



terminal commands

open terminal

type:

wsl

then to get into container

cd ~/dev/il-ur5e-checkers-sim

docker compose exec dev bash

Rebuild Docker Image:

cd ~/dev/il-ur5e-checkers-sim
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose exec dev bash

Run Gazebo Server:

gz sim -s -v 3 /opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/empty.sdf

Then in a new terminal spawn in the UR5e:

source /opt/ros/jazzy/setup.bash

source /workspaces/ur5e-checkers-irl/ros_ws/install/setup.bash

XACRO="$(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro"

xacro "$XACRO" ur_type:=ur5e name:=ur5e tf_prefix:="" > /tmp/ur5e.urdf

ros2 run ros_gz_sim create -name ur5e -string "$(cat /tmp/ur5e.urdf)" -x 0 -y 0 -z 0.2

Verify:

gz model --list
gz topic -e -t /world/empty/pose/info -n 1 | grep -E "name: \"ur5e\"|name: \"base_link\"" -n


