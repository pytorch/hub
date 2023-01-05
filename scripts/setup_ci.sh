#!/usr/bin/env bash
set -ex -o pipefail

# Set up NVIDIA docker repo. See https://nvidia.github.io/libnvidia-container/
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
         && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
         && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
               sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
               sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#   sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# ubuntu=ubuntu20.04
# curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# echo "deb https://nvidia.github.io/libnvidia-container/$ubuntu/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
# echo "deb https://nvidia.github.io/nvidia-container-runtime/$ubuntu/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
# echo "deb https://nvidia.github.io/nvidia-docker/$ubuntu/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list




# Remove unnecessary sources
sudo rm -f /etc/apt/sources.list.d/google-chrome.list
sudo rm -f /etc/apt/heroku.list
sudo rm -f /etc/apt/openjdk-r-ubuntu-ppa-xenial.list
sudo rm -f /etc/apt/partner.list
sudo rm -f /etc/apt/sources.list.d/nvidia-container-runtime.list*
sudo rm -f /etc/apt/sources.list.d/nvidia-docker.list*

curl https://cli-assets.heroku.com/apt/release.key | sudo apt-key add -

sudo apt-get -y update
# sudo apt-get -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
sudo apt-get -y remove --force-yes linux-image-generic linux-headers-generic linux-generic docker-ce
sudo apt autoremove
sudo apt-get -y install linux-headers-5.4.0-135-generic
sudo apt-get -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  nvidia-container-runtime \
  nvidia-docker2 \
  expect-dev

sudo systemctl start docker
sudo pkill -SIGHUP dockerd

DRIVER_FN="NVIDIA-Linux-x86_64-450.51.06.run"
wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
nvidia-smi
