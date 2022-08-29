# Training the model

To run the model, after cloning this project, it is recommended to create a virtual environment. Then, run the following commands.

## Install the necessary requirements

To run only the training / prediction pipelines:
```PowerShell
pip install -r requirements.txt
```

To run the notebooks, other libraries are needed. To install them, run the following

```PowerShell
pip install -r dev_requirements.txt
```

## To run the model

To train or use the  model, the dataset files must be in the directory listed in the `config.yml` file of the project package. The default configuration is `data/raw`. As already mentioned, this dataset isn't available in this project, due to Copyright restrictions.

**Training the model**

If the needed file are available, run following command the command.

```PowerShell
python run_app.py -c train
> saving model as: svc_classification_pipe.pkl
```

Where -c stands for the --command argument. The options for that argument are `train` or `predict`.

**Predicting results**

To predict the results for a given file, for example `4.csv`, run the following command. When this command is run and no persisted model is found, the training pipeline is executed automatically.

```PowerShell
python run_app.py -c predict -f data/raw/4.csv
> Making prediction with model persisted as 'svc_classification_pipe.pkl'. Prediction: Before
> Result: {'prediction': 'Before', 'model': 'svc_model'}
```

Where -f stands for the --file argument. It checks if the file exist before calling the function to run the pipeline.
