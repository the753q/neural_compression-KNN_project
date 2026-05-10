### Setup
Use the uv package manager for ideal compatibility.
Run `uv sync` to download the required libraries.

### Run
Training can be run using `uv run train.py`.
Evaluation can be performed using `uv run evaluate.py`.


### Structure
Project implementation is contained in the /src/ folder, with the main runnable files being the train.py and evaluate.py files.

Each model file is contained in its own file, in the /src/models/ directory.


### Get datasets
Downloads data and creates the datasets folder.
```
python /src/download_datasets.py
```

### Datasets used:
- https://www.kaggle.com/datasets/priyerana/imagenet-10k
- https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images
- https://www.kaggle.com/datasets/daehoyang/flickr2k
- https://www.kaggle.com/datasets/sqdartemy/minecraft-screenshots-dataset-with-features
- https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset