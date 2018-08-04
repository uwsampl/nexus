Build docker image
==================
To build nexus docker images, you need to install
[docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)(>=1.12) first.

There are four docker images that needs to be built. First you need to build the
base docker image that installs all dependent libraries required for Nexus.

Following docker build commands contain -t and -f options. -t option is followed 
by name and optionally a tag in the ‘name:tag’ format. -f option is followed by 
name of the Dockerfile (Default is ‘PATH/Dockerfile’).

```
$ cd nexus/dockerfiles
$ docker build -t nexus/base -f NexusBaseDockerfile .
```

Next we can build the docker images for backend, scheduler, and application
library respectively.
```
$ docker build -t nexus/backend -f NexusBackendDockerfile .
$ docker build -t nexus/scheduler -f NexusSchedulerDockerfile .
$ docker build -t nexus/applib -f NexusAppLibDockerfile .
```

Compile sample applications
---------------------------
In the Nexus repo, we provide a few sample applications located at `nexus/apps`.
In each application directory, you can build the application docker image by
```
$ docker build -t nexus/app_name -f Dockerfile .
```
