Anomaly Detection Pipeline on Azure Databricks
==============================

The following is an anomaly detection data pipeline on Azure Databricks. This solution was built to demonstrate how to build Advance Analytics Pipelines on Azure Databricks, with a particular focus on the Spark MLLib library. This solution includes:
1. Initial ETL Data loading process into SparkSQL tables
2. Model training and scoring
   - Explanation of Pipelines, Transformer and Estimators
   - Sample Custom Estimator (PCAAnomaly)
3. Persisting trained models
4. Productionizing models through
    -  Batch inference
    -  Streaming

# Architecture
![Architecture](images/archi.PNG?raw=true "Architecture")


# Data
[KDD Cup 1999 Data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

# Deployment

- Ensure you are in the root of the repository
- To deploy the solution, use one of the following commands:
    1. (*Easiest*) Using pre-built docker container: `docker run -it devlace/azdatabricksanomaly`
    2. Build and run the container locally: `make deploy_w_docker`
    3. Deploy using local environment (see requirements below): `make deploy`
- Follow the prompts to login to Azure, name of resource group, deployment location, etc.
- When prompted for a Databricks Host, enter the full name of your databricks workspace host, e.g. `https://southeastasia.azuredatabricks.net` 
- When prompted for a token, you can [generate a new token](https://docs.databricks.com/api/latest/authentication.html) in the databricks workspace.
  
To view additional make commands run `make`

## For local deployment

### Requirements

- [Azure CLI 2.0+](https://azure.github.io/projects/clis/)
- [Python virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) or [Anaconda](https://anaconda.org/anaconda/python)
- [jq tool](https://stedolan.github.io/jq/download/)
- Check the requirements.txt for list of necessary Python packages. (will be installed by `make requirements`)

### Development environment

- The following works with [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- Clone this repository
- `cd azure-databricks-anomaly`
- Create a python environment (Virtualenv or Conda). The following uses virtualenv.
    - `virtualenv .`  This creates a python virtual environment to work in.
    - `source bin/activate`  This activates the virtual environment.
- `make requirements`. This installs python dependencies in the virtual environment.

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── deploy             <- Deployment artifacts
    │   │
    │   └── databricks     <- Deployment artifacts in relation to the Databricks workspace
    │   │
    │   └── deploy.sh      <- Deployment script to deploy all Azure Resources
    │   │
    │   └── azuredeploy.json <- Azure ARM template w/ .parameters file
    │   │
    │   └── Dockerfile     <- Dockerfile for deployment
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Contains the powerpoint presentation, and other reference materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
