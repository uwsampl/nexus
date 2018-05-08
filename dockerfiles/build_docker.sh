set -e

docker build -t nexus/base -f NexusBaseDockerfile .
docker build -t nexus/backend -f NexusBackendDockerfile .
docker build --no-cache -t nexus/scheduler -f NexusSchedulerDockerfile .
docker build --no-cache -t nexus/applib -f NexusAppLibDockerfile .
