## Superpixel Sampling Networks


This is the code accompanying the **ECCV 2018** publication on **Superpixel Sampling Networks**.
Please visit the project [website](http://varunjampani.github.io/ssn) for more details about the paper and overall methodology.

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### Installation

#### Caffe Installation

1. Go to 'lib' folder if you are not already there:
```
cd ssn_superpixels/lib/
```

2. We make use of layers in 'Video Propagation Networks' caffe repository and add additional layers for SSN superpixels:
```
git clone https://github.com/varunjampani/video_prop_networks.git
```

3. Manually copy all the source files  (files in `lib/include` and `lib/src` folders)
to the corresponding locations in the `caffe` repository. In the `ssn_superpixels/lib` directory:
```
cp src/caffe/layers/* video_prop_networks/lib/caffe/src/caffe/layers/.
cp src/caffe/test/* video_prop_networks/lib/caffe/src/caffe/test/.
cp src/caffe/proto/caffe.proto video_prop_networks/lib/caffe/src/caffe/proto/caffe.proto
cp include/caffe/layers/* video_prop_networks/lib/caffe/include/caffe/layers/.
```

4. Install Caffe following the installation [instructions](http://caffe.berkeleyvision.org/installation.html).
In the `ssn_superpixels/lib` directory:
```
cd video_prop_networks/lib/caffe/
mkdir build
cd build
cmake ..
make -j
cd ../../../..
```

Note: If you install Caffe in some other folder, update `CAFFEDIR` in `config.py` accordingly.

#### Install a cython file

We use a cython script taken from 'scikit-image' for enforcing connectivity in superpixels. To compile this:

```
cd lib/cython/
python setup.py install --user
cd ../..
```

### Usage: BSDS segmentation

#### Data download

Download the BSDS dataset into `data` folder:
```
cd data
sh get_bsds.sh
cd ..
```

#### Superpixel computation

1. First download the trained segmentation models using the `get_models.sh` script in the `models` folder:
```
cd models
sh get_models.sh
cd ..
```

2. Use `compute_ssn_spixels.py` to compute superpixels on BSDS dataset:
```
python compute_ssn_spixels.py  --datatype TEST --n_spixels 100 --num_steps 10 --caffemodel ./models/ssn_bsds_model.caffemodel --result_dir ./bsds_100/
```

You can change the number of superpixels by changing the `n_spixels` argument above, and you can update the `datatype` to `TRAIN` or `VAL` to
compute superpixels on the corresponding data splits.

If you want to compute superpixels on other datasets, update `config.py` accordingly.

#### Evaluation

For superpixel evaluation, we use scripts from [here](https://github.com/wctu/SEAL) for computing ASA score and
scripts from [here](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) for computing
Precision-Recall and other evaluation metrics.

#### Training

Use `train_ssn.py` to train on BSDS training dataset:

```
python train_ssn.py --l_rate=0.0001 --num_steps=10
```

### Citation

Please consider citing the below paper if you make use of this work and/or the corresponding code:

```
@inproceedings{jampani18ssn,
	title = {Superpixel Samping Networks},
	author={Jampani, Varun and Sun, Deqing and Liu, Ming-Yu and Yang, Ming-Hsuan and Kautz, Jan},
	booktitle = {European Conference on Computer Vision (ECCV)},
	month = September,
	year = {2018}
}
```
