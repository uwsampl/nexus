set -e

docker build -t nexus/base -f NexusBaseDockerfile .
docker build -t nexus/backend -f NexusBackendDockerfile .
docker build -t nexus/scheduler -f NexusSchedulerDockerfile .
docker build -t nexus/applib -f NexusAppLibDockerfile .
