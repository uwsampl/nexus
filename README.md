Nexus
=====

[![Docker Image](https://img.shields.io/microbadger/image-size/abcdabcd987/nexus)](https://hub.docker.com/repository/docker/abcdabcd987/nexus)

Nexus is a scalable and efficient serving system for DNN applications on GPU
cluster.

## SOSP 2019 Paper

* Check out our SOSP 2019 paper [here](https://doi.org/10.1145/3341301.3359658).
* Check out the [Google Drive](https://drive.google.com/open?id=104UqrlNrfJoQnGdkxTQ56mfxSBFyJTcr) that contains a sample of video dataset.

## Building Nexus

See [BUILDING.md](BUILDING.md) for details.

## Docker and Examples

We provide a [Docker image](https://hub.docker.com/repository/docker/abcdabcd987/nexus)
so that you can try Nexus quickly. And there is an example that goes step by
step on how to run Nexus with a simple example application. We recommend you to
take a look [here](examples/README.md).

## Deployment

### Download Model Zoo

Nexus publishes public model zoo on our department-hosted GitLab. To download,
you need to install [Git LFS](https://git-lfs.github.com/) first. Then, run:

```bash
git clone https://gitlab.cs.washington.edu/syslab/nexus-models
cd nexus-models
git lfs checkout
```

### Run the Profiler

Nexus is a profile-based system. So before running Nexus, make sure you have
profiled all the GPUs. To profile a certain model on a certain GPU, run:

```bash
nexus/tools/profiler/profiler.py --gpu_list=GPU_INDEX --gpu_uuid \
    --framework=tensorflow --model=MODEL_NAME \
    --model_root=nexus-models/ --dataset=/path/to/datasets/
```

The profile will be saved to the `--model_root` directory.
See [examples](examples/README.md) for more concrete usage.

### Run Nexus

To run Nexus, you need to run the **scheduler** first, then spawn a **backend** for each
GPU card, and finally run the Nexus **frontend** of your application.
See [examples](examples/README.md) for more concrete usage.
