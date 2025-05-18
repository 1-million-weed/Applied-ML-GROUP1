import yaml
from f1_predictor.ml.pipeline import Pipeline

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def Formula1Predictor():
    config = load_config()
    model_config = config['model']
    dataset_config = config["dataset"]
    training_config = config["training"]
    test_config = config["testing"]
    inference_config = config["inference"]
    pipeline = Pipeline(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        test_config=test_config,
        inference_config=inference_config,
    )
    pipeline.run()


if __name__ == '__main__':
    Formula1Predictor()
    
