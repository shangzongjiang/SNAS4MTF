# SNAS4MTF
This repo is the official implementation for Scale-Aware Neural Architecture Search for Multivariate Time Series Forecasting

## 1.1 The framework of SNAS4MTF
 ![framework](https://user-images.githubusercontent.com/18440709/138592754-39a1f4d0-0f9e-4430-96cc-bf74c95f557c.png)

# Prerequisites
* Python 3.6.12
* PyTorch 1.2.0
* math, sklearn, numpy
# Datasets
## 1.1 METR-LA
The raw data is in http://pems.dot.ca.gov. This dataset is collected by the Los Angeles Metropolitan Transportation Authority and contains the average traffic speed measured by 207 loop detectors on the highways of Los Angeles County between March 2012 and June 2012.
## 1.2 PEMS-BAY
The raw data is in.This dataset is collected by California Transportation Agencies and contains the average traffic speed measured by 325 sensors in the Bay Area between January 2017 and May 2017.
# Running
1.Install all dependencies listed in prerequisites

2.Download the dataset

3.Neural Architecture Search
 python search.py --config config/PEMS_BAY_para.yaml |& tee logs/search_PEMS_BAY.log
 python search.py --config config/METR_LA_para.yaml |& tee logs/search_METR_LA.log

4.Training
python train.py --config config/PEMS_BAY_para.yaml  |& tee logs/train_PEMS_BAY.log
python train.py --config config/METR_LA_para.yaml |& tee logs/train_METR_LA.log

5.Evaluating
python test.py --config config/PEMS_BAY_para.yaml |& tee logs/test_PEMS_BAY.log
python test.py --config config/METR_LA_para.yaml |& tee logs/test_METR_LA.log

## Citation
Please cite the following paper if you use the code in your work:
```
@Inproceedings{616B,
  title={Scale-Aware Neural Architecture Search for Multivariate Time Series Forecasting.},
  author={Donghui Chen, Ling Chen, Youdong Zhang, et al.},
  booktitle={},
  year={2021}
}
```
