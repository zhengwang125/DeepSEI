# DeepSEI

This is the implementation of our paper "On Inferring User Socioeconomic Status with Mobility Records" (BigData22).

## Requirements

* Python >= 3.5 (Anaconda3 is recommended and 3.7 is tested)
* Tensorflow & scikit-learn (2.3.1 and 1.0.2 are tested)
* scikit-mobility 

Please refer to the source code to install the required packages that have not been installed in your environment such as scikit-mobility. You can install these packages with pip in a shell as

```
pip install scikit-mobility
```

## Dataset & Preprocessing

Download & unzip the dataset [Geolife](http://research.microsoft.com/en-us/downloads/b16d359d-d164-469e-9fd4-daa38f2b2e13/) to the folder `./Data`. 

### Indicators
Run [`features/indicators.py`](features/indicators.py), then the trajectories are separated by user number and stored into the directory `./user` (e.g., `./user/000`); additionally, the basic indicators of the users' features as well as their stay points are obtained; then the stay points are also seperated weekly with the format "user/weeks/userno" (e.g., `user/weeks/000`) while the features are summed up and dumped in a binary file with the name `user/weeks/feature_in_bj`.
```
python indicators.py
```

Run [`Vor_prices.py`](Vor_prices.py) to obtain the economy information (house price) of the whole Beijing city, which is adopted in [`get_inputs.py`](./get_inputs.py) to gain the label of each item in our set.
```
python Vor_prices.py
```

### Input data
Run [`get_inputs.py`](./get_inputs.py) to get the input vectors and the labels, which are stored in the directory `user/input`. You can change the parameter cellsize here to gain a better performance.
```
python get_inputs.py 
```

## Running Procedures

### Pre-training
Run [`radius_pretrain.py`](radius_pretrain.py) and [`entropy_pretrain.py`](entropy_pretrain.py) to pretrain the embedding vectors with Skip-gram model, and the checkpoints are stored in `./radius_pre` and `./entropy_pre`, which will be loaded by the static network. The directories of `radius_pre` and `entropy_pre` will be generated automatically by running this script.

Run [`Deep_net.py`](Deep_net.py) to pretrain the static network, and the checkpoints are stored in `./ckpt_deep_i` (i is the number of classes of the label ranging from 2 to 5) , which will be loaded by DeepSEI model. The directory of `ckpt_deep_i` will also be generated automatically.
```
python radius_pretrain.py
python entropy_pretrain.py
python Deep_net.py
```

### Classification
Run [`DeepSEI.py`](DeepSEI.py) to train and test our classification model: DeepSEI. The generated models will be stored in the folder `./ckpt_i` (i is the number of classes of the label ranging from 2 to 5) automatically, and you can pick one model with the best performance on the validation data as your model from them. The parameter of spatiality, temporality and activity granularity here. More importantly, 
the number of classes can be changed here to perform different tasks. The history of the training process is recorded in the directory `./logs`, which can be displayed in TensorBoard. In addition, the variable `history` in function `train` also records the training process of each epoch (e.g., loss, accuracy, precision and recall). The training time and f1 score of each epoch are added in callback functions, and later printed.
```
python DeepSEI.py
```

### Clustering
Run [`Clustering.py`](Clustering.py) to do the task of clustering. The inputs of our clustering model are the output vectors of one hidden layers in the classification model, so the clustering task has to be run after obtaining the checkpoints of classification model. The number of clusters is also changable according to the classification model ranging from 2 to 5.
```
python Clustering.py
```

