import atexit
import yaml
import logging.config
import logging.handlers
import pathlib
import queue

from f1_predictor.ml.pipeline import Pipeline
from mylogger import MyJSONFormatter

logger = logging.getLogger(__name__)


def setup_logging():
    pathlib.Path("logs").mkdir(exist_ok=True)

    simple_formatter = logging.Formatter(
        "%(asctime)s: %(name)s (%(levelname)s) - %(message)s"
    )

    json_formatter = MyJSONFormatter(
        fmt_keys={
            "level": "levelname",
            "logger": "name",
            "module": "module",
            "function": "funcName",
            "line": "lineno",
            "thread_name": "threadName"
        }
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(simple_formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/f1_predictor.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        # encoding="utf-8" # Dont really want this rn
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)

    log_queue = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)

    listener = logging.handlers.QueueListener(
        log_queue,
        stderr_handler,
        file_handler,
    )

    listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    atexit.register(listener.stop)

    logger.info("Logging setup complete")

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
    logger.info("Starting Formula 1 Predictor")
    config = load_config()
    model_config = config['model']
    dataset_config = config["dataset"]
    training_config = config["training"]
    eval_config = config["evaluation"]
    inference_config = config["inference"]
    pipeline = Pipeline(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        eval_config=eval_config,
        inference_config=inference_config,
    )
    pipeline.run()

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

if __name__ == '__main__':
    setup_logging()
    Formula1Predictor()
