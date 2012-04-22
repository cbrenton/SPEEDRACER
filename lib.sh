#!/bin/bash

cd ./lib/pngwriter-0.5.4
make DESTDIR=../pngwriter
make install
cd ../../src
if [ ! -e main.cu ] && [ ! -h main.cu ]
then
   ln -s ./main.cpp ./main.cu
fi
