# Dataset setup
Here you can place the datasets downloaded, convert them into temporal or other augmentations.

For synthetic data, you can drop the `synthetic_reasoning_dataset` here.

UCF-Crime can be downloaded from the official github: https://github.com/Xuange923/Surveillance-Video-Understanding
Or from Kaggle: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset 

Then converted into a minimal versions with only the main actions by using the test temporal annoations. Use the script:
`python temporal_dataset.py --annotations temporal_annotations/test.json --video-root videos/ --output-root videos_temporal`

RWF2000 can be downloaded from Kaggle: https://www.kaggle.com/datasets/vulamnguyen/rwf2000

Privacy filters can be applied by using the simple YOLO-based script `add_privacy_filter.py` or using the GAN-based deep_privacy2 framework.

Clone the deep_privacy2 framework https://github.com/hukkelas/deep_privacy2 and replace the anonymize.py file with the one in this project.
