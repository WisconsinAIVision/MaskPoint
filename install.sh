#!/usr/bin/env sh
HOME=`pwd`

cd $HOME/extensions/pointnet2
python setup.py install

cd $HOME/extensions/pointops
python setup.py install
