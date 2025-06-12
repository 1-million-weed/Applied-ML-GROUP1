# 🏎️ Formula 1 Predictor

Welcome to our Formula 1 Predictor project! This project was made for the Applied Machine Learning course at the University of Groningen, part of the Artificial Intelligence Bachelor's program. 

The goal is to predict the outcome of Formula 1 races leveraging machine learning. 

## Getting Started

Currently, our API is not publicly available, but you can still run the code locally to train and test our diverse selection of models.

All changeable parameters are stored in the `config.py` file, so you can easily adjust them to your needs.

### Running the Project Locally

> [WARNING] This project is built with Python 3.9. Make sure you have this version installed on your machine. If you are using a different version, you may encounter compatibility issues.

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/1-million-weed/Applied-ML-GROUP1.git
   cd Applied-ML-GROUP1
   ```
2. Set up a virtual environment (our team used a mix between `pipenv` and `venv`):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### OR 

2. If you prefer using `pipenv`, you can install the dependencies with:
   ```bash
   pip install pipenv
   pipenv install
   ```
   Ensure you are in the directory where the `Pipfile` is located when running this.
3. To run just the API server, execute:
   ```bash
   python main.py
   ```
   Then head over to 'http://localhost:8000/docs' to access the Swagger UI and test the API endpoints.
4. To run the Streamlit application, execute:
   ```bash
   streamlit run main.py
   ```
   Then head over to 'http://localhost:8501' to acces our streamlit api demo.

## THE CONFIG FILE

Everything in this project is configurable through the `config.py` file. This includes:
- The `model` to be used for predictions
- Dataset **acquisition**
  - Whether to acquire new data or use existing data
  - Whether to `preprocess` the data
  - Whether to `generate` new features from raw data
  - Training and testing data `split`
- Model **training**
  - Whether to `train` a new model or use an existing one
  - `Tensorboard` logging
- Model **evaluation**
  - Whether to `evaluate` the model on the test set
  - Whether to show the evaluation `plots`
- Model **inference**
  - Whether to run `inference` on the model
  - Whether to activate the `API server`
  - Whether to run the `Streamlit` application
- **Logging**
  - The level of logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)

## API Endpoints

The API provides several endpoints to interact with the model:
- **GET /**: API Health check
- **GET /meetings**: List of all meetings from the current season
- **GET /meetings/{meeting_id}/max-laps**: List the maximum amount of laps in a race
- **GET /docs**: Swagger UI documentation

- **POST /predict**: Predict the outcome


## Notable references:

- [Elo calculation code](https://www.kaggle.com/code/lorenzojayd/elo-system-in-formula-1/notebook)

## Project Structure

```bash
├───data  # Stores raw .csv files
├───models  # Stores .pkl files for trained models
├───experimental  # Contains experimental .ipynbs & .py
├───f1_predictor
│   ├───app # Contains the Streamlit app
│   ├───data  # stores processed .csv files
│   ├───data_acquisition # For acquiring data from the FastF1 API for 2025 data
│   ├───features # For scripts and logic for feature engineering
│   ├───ml # Contains the machine learning logic (pipelines & managers)
│   └───models  # For model creation, not storing .pkl
├───reports # For outputs and visualisations
├───tests
│   ├───data
│   ├───features
│   └───models
├───.dockerignore
├───.gitignore
├───.pre-commit-config.yaml
├───config.py
├───Dockerfile
├───main.py
├───mylogger.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
├───requirements.txt
```
