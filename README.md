# Multi-Source Domain Adaptation

# Usage:
To start with this project, please get the dataset first.

### Dataset
For users that have wget, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `data`.  If you are using other operating systems, you should download the dataset from [this link](https://doc-08-ao-docs.googleusercontent.com/docs/securesc/qkfagn1062bdd1taiojf0vjvgkk73n67/glf5lp7gdo4nisg93hlu85kch3rgi5vk/1561176000000/10280900870256151746/07847013239326180463/1S8zc21EthjpCg7ckslAD4PfovWyErQ0p?e=download) and unzip the compressed file manually and put them in the `data` folder.

### Download Pretrained Models
If you just want to run and predict, please use the following command. Otherwise, skip this step.

    bash ./get_model.sh
The shell script will download the models trained for 4 cases and store in the folder `models/`. If you are having trouble downloading the models using shell script, you can download the models from [real_model](https://drive.google.com/open?id=1YHgXQA5bMcAjM0mlsEk9eeQ6gi7YybZH) , [infograph_model](https://drive.google.com/open?id=1ixws3BRVLY7lX1xYVwHocj5_uJl8-bzA) , [quickdraw_model](https://drive.google.com/open?id=1DpiTrBUgPMSOc4d92PCuTOszpG2H7kff) , [sketch_model](https://drive.google.com/open?id=19dK0EnJvEvFXgepI98FfK9JFW52VphhG).


### Train the iDANN Model
Simply type the following command.

    python3 train_weights_DANN.py
There are four different case for training, if you want to switch other training case, you can alter the following lines in the code:
```python
source1 = 'sketch'
source2 = 'infograph'
source3 = 'real'
target = 'quickdraw'
```

### Prediction
Shell script to run predcition.

    bash ./predict.sh $1 $2 $3
 - `$1` is the filename of your output prediction file (e.g. `test_pred.csv`)
 - `$2` is target domain (e.g. `real`,`infograph`,`quickdraw`,`sketch`)
 - `$3` is your model path (e.g. `./models/real.pth`,`./models/infograph.pth`,`./models/quickdraw.pth`,`./models/sketch.pth`)

 I deafult use `adaptive_weight=True` for pretained model, you can change to False in test.py to get average weight result.
 To get kaggle result, please run `bash ./predict.sh $1 real ./models/real.pth` after you download pretained model.

 | adaptive_weight | kaggle public score | kaggle private score | info | qkr | skt |
 | :--: | :--: | :--: | :--: | :--: | :--: |
 | True | 0.59309 | 0.59051 | 0.2130 | 0.1551 | 0.4500 |
 | False | 0.59794 | 0.59430 | 0.2096 | 0.1521 | 0.4428 |

If you want to use your own model, just note that `predict.sh` is only for iDANN Model
### Train the M3SDA-beta Model
Befor training, you must specify the target.   

    python3 train_beta.py --target $1
 - `$1` is the name of target domain.

You can adjust batch_size, epoches, and use a pretrained model like

    python3 train_beta.py --target $1 --bs $2 --ep $3 --pretrain $4
 - `$1` is the name of target domain.  
 - `$2` is the number of batch_size.  
 - `$3` is the number of epoches.  
 - `$4` is the path of the model.  


## Packages
Packages used to implement this project:

> [`CUDA`](https://www.h5py.org/https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64): 9.0  
> [`python`](https://www.python.org/): 3.7.3  
> [`torch`](https://pytorch.org/): 1.0.1  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.2  
> [`PIL`](https://pypi.org/project/Pillow/): 5.4.1  
> [`torchvision`](https://pypi.org/project/torchvision/): 0.2.2  
> [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/)   
> [The Python Standard Library](https://docs.python.org/3/library/)
