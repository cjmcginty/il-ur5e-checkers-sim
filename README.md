# il-ur5e-checkers-sim



terminal commands

//open terminal

//type:

wsl

//then to get into container

cd ~/dev/il-ur5e-checkers-sim

docker compose exec dev bash

Rebuild Docker Immage:

cd ~/dev/il-ur5e-checkers-sim
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose exec dev bash
