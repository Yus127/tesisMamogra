# Data folder

This folder should contain one or more datasets. The dataset folder should contain the datapoints and jsons files with the dataset information. If the dataset contains a train.json file, this file will be used to generate training and validation datasets. If the dataset contains a test.json file, this file will be used to generate the test dataset.

The train.json and test.json file are expected to have the following format:

```json
[
  {
    "filename": "Name of the file",
    "image_path": "Path to file from .data/",
    "report": "Medical report."
  }
]
```
