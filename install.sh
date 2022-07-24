#!/usr/bin/env sh
HOME=`pwd`

pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

cd $HOME/extensions/pointops
python setup.py install
