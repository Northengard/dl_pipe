# Deep learning pipe

NN Training pipeline for [dfcd task](https://www.kaggle.com/c/deepfake-detection-challenge/overview).

## Prepare to use:
  * git clone https://github.com/Northengard/dl_pipe
  * cd dl_pipe
  * pip install requirements.txt  
 
## Usage:
  * create your config file for training. See example in configs.
  * insert your own model and data_set if necessary
  * python train.py --config <Your_config_file.yaml>
  
You can follow an example of "ForwardRegression model" and DFC dataset

Pretrained weights are [available](https://drive.google.com/open?id=1xXj_GmgETHq-S1OTEQZY8JX_9dBxAX2_). This snapshot has ~86 accuracy on validation set
