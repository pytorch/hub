#!/usr/bin/env bash
set -ex -o pipefail

# Set up NVIDIA docker repo
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
echo "deb https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list

# Remove unnecessary sources
sudo rm -f /etc/apt/sources.list.d/google-chrome.list
sudo rm -f /etc/apt/heroku.list
sudo rm -f /etc/apt/openjdk-r-ubuntu-ppa-xenial.list
sudo rm -f /etc/apt/partner.list

sudo apt-get -y update
sudo apt-get -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
sudo apt-get -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  docker-ce=5:18.09.4~3-0~ubuntu-xenial \
  nvidia-container-runtime=2.0.0+docker18.09.4-1 \
  nvidia-docker2=2.0.3+docker18.09.4-1 \
  expect-dev

sudo pkill -SIGHUP dockerd

DRIVER_FN="NVIDIA-Linux-x86_64-450.51.06.run"
wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
nvidia-smi
