# Training Models

The files in this directory contain all the code required to train/retrain the models used in Home Credit default 
prediction service. Note that this assumes that you already have the data required to retrain the model and it is stored
in the `VirtualDataWarehouse` directory within the root of the project.

## Credit Default Model
Assuming that you follow the instructions in the README file. 
To retrain/train the credit default model, run the following commands in the project root folder:
* Run `./fire rebuild`
* Run `./fire shell`
* Run `cd model-retraining`
* Run `python3 DoETL.py`
* Run `python3 train_model.py`

This will generate a model binary that will be stored inside `HomeCreditDefaultPrediction/prediction/model-binaries/`
The model binary will contain metadata in the form of a dictionary that has the following:
  - A brief description of the model
  - Timestamp for when the model was built
  - A sha256 hash identifying specifying the model version.

Please note that by default the existing model will be moved into `HomeCreditDefaultPrediction/prediction/model-binaries/archive_binaries`

