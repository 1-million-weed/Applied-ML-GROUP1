import yaml
from f1_predictor.ml.pipeline import Pipeline

def load_config(path="config.yaml") -> dict:
    """
    Load a YAML configuration file for model, dataset, and training settings.

    :param path: Path to the configuration YAML file. Defaults to "config.yaml".
    :type path: str, optional
    :return: Parsed configuration dictionary.
    :rtype: dict
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def Formula1Predictor():
    """
    Initialize and run the Formula 1 prediction pipeline based on configuration.

    Loads model, dataset, training, testing, and inference configurations,
    initializes the pipeline, and runs it.
    """
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
    
