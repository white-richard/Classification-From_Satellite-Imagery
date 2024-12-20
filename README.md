# Classification-From-Satellite-Imagery

Classifying aircraft from satellite imagery using the HRPlanes Dataset. The goal is to identify aircraft and determine where images of the ground may be obscured by flying aircraft.

## Prerequisites

* Run all commands in the terminal from the project directory (`Classification-From-Satellite-Imagery/`):

* Before running the project in the terminal, ensure relative file path compatibility by executing the following command:

```bash
export PYTHONPATH=$(pwd)
```

* Install pip dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Model Predictions

To inference on the model:

1. Add images to be inferenced in input directory (`data/inference_input/`)
1. Run the main script using the inference flag:

```bash
python3 src/main.py --inference
```

### Model Training

To train models:

1. Run the main script using the train and test flag:

```bash
python3 src/main.py --train --test
```

## Directory Structure
- **src/**: Contains the source code for evaluation and training scripts.
- **output/**: Stores output images, plots, and other evaluation results.

## Dependencies
Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

## Citation
If you use the HRPlanesv2 dataset in your research or project, please cite the following:

"Unsal, Dilsad. (2022). HRPlanesv2 - High Resolution Satellite Imagery for Aircraft Detection [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7331974"