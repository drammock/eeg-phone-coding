#!/bin/sh

# install htop
sudo apt install htop

# mount EBS drive
sudo mkfs -t ext4 /dev/xvdf
MOUNTPOINT="$HOME/ebsdrive"
mkdir "$MOUNTPOINT"
sudo mount /dev/xvdf "$MOUNTPOINT"
sudo chown ubuntu:ubuntu $MOUNTPOINT

# install miniconda
PREFIX="$MOUNTPOINT/miniconda"
MINICONDA_PATH="$PREFIX/bin"
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $PREFIX
export PATH="$MINICONDA_PATH:$PATH"
conda env create -f params/conda-env.yaml
source activate eeg_phone_coding

# make target directory for rsync
mkdir "$MOUNTPOINT/eeg_phone_coding"
