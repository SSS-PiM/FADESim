#!/bin/bash
data_dir=data

if [ ! -d "$data_dir" ]; then
    mkdir $data_dir
fi

cd $data_dir
# # mnist dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# # cifar10 dataset
# wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# scientific computing dataset, matrix market format, suitesparse matrix collection
# HB/nos1 237*237, 1017 nonzeros, symmetric, positive definite
# wget https://suitesparse-collection-website.herokuapp.com/MM/HB/nos1.tar.gz
# Pothen/sphere3 258*258, 1794 nonzeros, symmetric
# wget https://suitesparse-collection-website.herokuapp.com/MM/Pothen/sphere3.tar.gz
# Muite/Chebyshev1 261*261, 2319 nonzeros
# wget https://suitesparse-collection-website.herokuapp.com/MM/Muite/Chebyshev1.tar.gz
# HB/bcsstk22, 138*138, 696 nonzeros, symmetric, positive definite
# wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk22.tar.gz

# unzip dataset
gunzip *.gz
# tar -xvf cifar-10-binary.tar
#tar -xvf nos1.tar.gz
#tar -xvf sphere3.tar.gz
#tar -xvf Chebyshev1.tar.gz
#tar -xvf bcsstk22.tar.gz
