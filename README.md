
#CS5824 Project Report:Learning to Answer Questions with Human-Like Responses

## Steps to Run this code

1. Download the dataset 

The dataset (ver. 1.0) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh?usp=sharing). 

Place it inside the folder with the name of `Data`

2. install the Required libraries

`pip3 install -r requirements.txt`

3. To train the models run the script and modify parameters as required. 

`sh train_all.sh 0`


This script will train 4 models on java and python datasets. Available models

roberta-base
microsoft/CodeBERT
Salesforce/codet5-base
uclanlp/plbart-base


here the 0 signifies the gpu is availble 1 would mean cpu compute.

4. To evaluate a model on java set use, update the model name as required 

`sh java_test_script.sh 0` 

5. To evaluate a model on python set use the following, update the model name as required. 

`sh python_test_script.sh 0`



