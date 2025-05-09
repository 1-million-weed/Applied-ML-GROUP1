from f1_predictor.models.train_model import Modeltrainer

if __name__ == "__main__":
    trainer = Modeltrainer('XGBRegressor')
    trainer.train()