from f1_predictor.models.test_model import ModelTester

if __name__ == "__main__":
    tester = ModelTester('RandomForestClassifier')
    tester.test()