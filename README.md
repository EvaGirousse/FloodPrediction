# Flood Forecasting

This is the code for EJ Team for the Flood Map Forecasting Hackathon.

## Requirements ##
The following packages are necessary to run the code:

* PyTorch
* Xarray
* Pandas
* Numpy
* sklearn
* matplotlib (for the notebook)

We provide a ```requirements.txt``` file for the package versions but it might be slightly incomplete.

## Quick Start ##
The main function to run the training and create the submission file can be found in the ```main_archi.py``` file. Remove the ```### TRAINING ###``` part if you just want to run the inference.

## Data

The data should be gathered in a ```data``` repository. Our model cuts the full resolution map into patches of size $p \times p$ and process all the points from the patches at the same time. The patches are created and saved thanks to the [```FloodDataProcessor```](src/dataset.py) class, and save in the directory: ```data/patch_p```, with $p$ the patch size. The model we are giving to you used patches of size 4, the patches are already created and saved into the ```data/patch_4``` repository. If you want to change the patch size, the ```FloodDataProcessor``` will create a new repository.

## Configuration ##

The code is parametrized by the ```config``` dictionnary which  can be found in the [```config.py```](src/config.py) file. You can change the ```Dates``` for the train and the test period. You can also change the files for the temperature (```t2mFile```) and the total precipitation (```tpFile```).

## Data Visualization ##

We also provide a notebook [results_analysis.ipynb](results_analysis.ipynb). You can see how we generate predictions on a validation set and find the plots we used for the report.

## Authors ##

Julie Keisler
Eva Girousse
