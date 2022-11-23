#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
cp cifar10.py cifar-10-batches-py/
cd cifar-10-batches-py
python3 cifar10.py
mkdir cifar10
tar -zcvf cifar10.tar.gz cifar10
