# Implementation of ITFR
Source code for our paper: **Intersectional Two-sided Fairness in Recommendation**, accepted by WWW 2024.

## Usage
### Requirements
First install the environment dependencies using the following command.
```
conda install --file requirements.txt
```
### C++ Evaluator
C++ code is used to output accuracy-based metrics during training, as used in [LightGCN](https://github.com/kuandeng/LightGCN). Thanks for their code! It needs to be compiled first using the following command:
```
python setup.py build_ext --inplace
```
### Download Datasets and Preprocess (Optional)
We provide the processed data in the ```data``` directory, and you can also choose to re-download the three datasets: [Tenrec](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html), [Movielens](https://grouplens.org/datasets/movielens/1m/), [LastFM](http://www.cp.jku.at/datasets/LFM-2b/), and run the following command to regenerate the data (note that the raw data path in the preprocessing code need to be modified):
```
cd preprocess
python tenrec_qba.py
python ml1m.py
python lfm2b.py
```
### Run the Shell
Run the shell ```run.sh``` to reproduce the results of ITFR on the three datasets.

## Citation
If you find this work is helpful to your research, please consider citing our paper:

```
@inproceedings{10.1145/3589334.3645518,
author = {Wang, Yifan and Sun, Peijie and Ma, Weizhi and Zhang, Min and Zhang, Yuan and Jiang, Peng and Ma, Shaoping},
title = {Intersectional Two-sided Fairness in Recommendation},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645518},
doi = {10.1145/3589334.3645518},
numpages = {12},
series = {WWW '24}
}
```

## Contaction
Please feel free to contact the author at yf-wang21@mails.tsinghua.edu.cn for any help.
