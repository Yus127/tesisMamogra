# Data folder

This folder should contain one or more datasets. The dataset folder should contain the datapoints and jsons files with the dataset information. If the dataset contains a train.json file, this file will be used to generate training and validation datasets. If the dataset contains a test.json file, this file will be used to generate the test dataset.

The train.json and test.json file are expected to have the following format:

```json
[
  {"filename": <file_name>,
   "image_path": <path to image from .data/>,
   "report": <Medical report.>
  }
]
```
