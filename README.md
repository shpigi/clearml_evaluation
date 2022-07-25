# clearml_evaluation
try out clearml

Get nvidia containers
see: https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia&f=2
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
apt update
apt -y install nvidia-container-toolkit
systemctl restart docker
```

Build the env container:
```bash
docker build -t clearml_env  .
```