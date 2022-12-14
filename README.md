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
Run [`features/indicators.py`](features/indicators.py), then the trajectories are separated by user number and stored into the directory `./user` (e.g., `./user/000`); additionally, the basic indicators of the users' features as well as their stay points are obtained; then the stay points are also seperated weekly with the format "user/weeks/userno" (e.g., `user/weeks/000`) while the features are summed up and dumped in a binary file with the name `user/weeks/feature_in_bj`. You can change the parameter stay point duration (MINS) and cellsize (CELLSIZE) here to gain a better performance.
```
python indicators.py
```

Run [`Vor_prices.py`](Vor_prices.py) to obtain the economy information (house price) of the whole Beijing city, which is adopted in [`get_inputs.py`](./get_inputs.py) to gain the label of each item in our set.
```
python Vor_prices.py
```

### Input data
Run [`get_inputs.py`](./get_inputs.py) to get the input vectors and the labels, which are stored in the directory `user/input`.
```
python get_inputs.py 
```

## Running Procedures

### Pre-training
Run [`spatial_pretrain.py`](spatial_pretrain.py) and [`temporal_activity_pretrain.py`](temporal_activity_pretrain.py) to pretrain the embedding vectors with Skip-gram model, and the checkpoints are stored in `./spatial_pre` and `./temporal_activity_pre`, which will be loaded by the deep network. The directories of `spatial_pre` and `temporal_activity_pre` will be generated automatically by running these scripts.

Run [`deep_net.py`](deep_net.py) to pretrain the deep network, and the checkpoints are stored in `./ckpt_deep_i` (i is the number of classes of the label ranging from 2 to 5) , which will be loaded by DeepSEI model. The directory of `ckpt_deep_i` will also be generated automatically.
```
python spatial_pretrain.py
python temporal_activity_pretrain.py
python deep_net.py
```

### Training
Run [`DeepSEI.py`](DeepSEI.py) to train and test our classification model: DeepSEI. The checkpoints of generated models will be stored in the folder `./ckpt_i` (i is the number of classes of the label ranging from 2 to 5) automatically, and the checkpoints of each epoch are stored with the loss values as their filenames. The number of classes can be changed here to perform different tasks. The history of the training process is recorded in the directory `./logs`, which can be displayed in TensorBoard. In addition, the variable `history` in function `train` also records the training process of each epoch (e.g., loss, accuracy, precision and recall). The training time and f1 score of each epoch are added in callback functions, and later printed.
```
python DeepSEI.py
```
### Classification
The models are trained and stored in the Training process, and can be loaded by the function `make_or_restore_model()` in the script [`DeepSEI.py`](DeepSEI.py). This function adopts the latest created checkpoints to restore the model for classification, and you may also pick one with the best performance on the validation data (e.g., the model with the lowest loss) as your model from them. We provide a function `classification(x)` in the script [`DeepSEI.py`](DeepSEI.py) with x as the input data, which loads the model and outputs the classification results.

### Clustering
Run [`Clustering.py`](Clustering.py) to do the task of clustering. The inputs of our clustering model are the output vectors of one hidden layers in the classification model, so the clustering task has to be run after obtaining the checkpoints of classification model. The number of clusters is also changable according to the classification model ranging from 2 to 5.
```
python Clustering.py
```

