#!/bin/sh

sudo su
yum groupinstall 'Development Tools'
yum install python3-devel
yum install atlas-sse3-devel lapack-devel
exit

curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda env create -f conda-env.yaml
source activate eeg_phone_coding

