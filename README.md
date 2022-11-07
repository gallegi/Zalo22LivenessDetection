# 1. About the project:
- Basic solution for Zalo AI Challenge 2022, track "Liveness Detection": https://challenge.zalo.ai/portal/liveness-detection
- Current approach: 2D image classification, not utilizing sequential characteristic of videos
- Main dependencies:
  - Pytorch lightning
  - Timm
  - Comet-ml

## 2. Overview
- The project is built upon Pytorch Lightning for training code organization, and Commet ML for expermiment tracking.
![comet ml log](/pictures/comet_log.png "Comet API key")

# 3. How to run:
## 3.0. Preparation
### a. Change comet-ml api key
- You need to use your own comet-ml api key
  - Step 1: Go to https://www.comet.com/ and register for new accont
  - Step 2: Go to https://www.comet.com/account-settings/apiKeys to copy api key
  - Step 3: Replace the "comet_api_key" field in config_v1.py with your copied value
  ![comet ml log](/pictures/get_comet_api_key.png "Comet API key")
Then when training you will be able to see experiment log in your Comet-ml Dashboard
### b. Prepare the data
- Unzip Liveness Detection data (train.zip and public_test.zip) inside data/ folder
- The structure should look like this
  - ----- data
  - --------|-- train
  - --------------|-- videos
  - --------------|-- label.csv
## 3.1. Train:
````
    python train.py
````
## 3.2. Evaluate:
````
    python evaluate.py --fold <fold num> --weight <path-to-trained-weight>
````
## 3.2. Predict:
````
    python predict.py --weight <path-to-trained-weight>
````