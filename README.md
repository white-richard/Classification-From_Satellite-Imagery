# Classification-From-Satellite-Imagery

Classifying aircraft from satellite imagery using the HRPlanes Dataset. The goal is to identify aircraft and determine where images of the ground may be obscured by flying aircraft.

---

## Usage

### Model Predictions

To run inference on the model:

1. Add images to be processed in the `data/inference_input/` directory.
2. Use the provided command-line tool to run inference:

```bash
run_inference
```
* Output Images: Results will be saved in the (`output/inference/`) directory.

### Model Training

To train the model:

1. Use the command-line tool:

```bash
train_model
```

2. Optionally, run the evaluation after training:

```bash
evaluate_model
```
* Output Images: Results will be saved in the (`output/test/`) directory.

## Directory Structure
* data/inference_input/: Directory for input images for inference.
* output/: Stores output images and evaluation JSON.
* config.py: Contains hyperparameters and configurations to tune the model.

## Dependencies
Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

## Prerequisites

* Run all commands in the terminal from the project directory (`Classification-From-Satellite-Imagery/`):

* Install the project in editable mode to set it up as a Python package:

```bash
pip install -e .
```

## Future Work

Future improvements will focus on addressing the model's limitations, such as detecting aircraft at higher altitudes. To achieve this, the RarePlanes dataset will be incorporated to enhance performance and robustness.

## Citation
If you use the HRPlanesv2 dataset in your research or project, please cite:

"Unsal, Dilsad. (2022). HRPlanesv2 - High Resolution Satellite Imagery for Aircraft Detection [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7331974"

If you use the RarePlanes dataset in your research or project, please cite:

CosmiQ Works & AI.Reverie. (2020). RarePlanes: A dataset of real and synthetic overhead imagery for object detection. Retrieved from [https://github.com/VisionSystemsInc/RarePlanes](https://github.com/VisionSystemsInc/RarePlanes)
