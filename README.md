## Training Setup

###
```
git clone git@github.com:ithemal/Ithemal.git
git clone git@github.com:ithemal/bhive.git
cd Ithemal/data_collection
## Build DynamoRIO for tokenizer
git clone --recurse-submodules -j4 https://github.com/DynamoRIO/dynamorio.git
cd dynamorio && mkdir build && cd build
cmake ..
make -j

# Go back to build Ithemal Tokenizer
cd Ithemal/data_collection
mkdir build && cd build
cmake -DDynamoRIO_DIR=../dynamorio/build/cmake ..

# Go Root to train
mkdir -p ./my_models
ITHEMAL_HOME=./Ithemal python3 train.py --epochs 20 --batch-size 64 --validation-split 0.15 --save-path ./my_models
```