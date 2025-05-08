import os
import pickle

class Modelmanager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.model_dir = os.path.join(os.path.dirname(self.parent_dir), "models")

    def save_model(self, model):
        """
        Save the model to a file.
        :param model: The model to save.
        """
        #make sure the model doesnt already exist
        if self.check_if_model_exists():
            self.delete_model()
        with open(os.path.join(self.model_dir, self.model_name), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self):
        """
        Load the model from a file.
        :return: The loaded model.
        """
        #make sure the model exists
        if not self.check_if_model_exists():
            raise FileNotFoundError(f"Model {self.model_name} not found in {self.model_dir}")
        with open(os.path.join(self.model_dir, self.model_name), 'rb') as f:
            model = pickle.load(f)
        return model
        
    def delete_model(self):
        """
        Delete the model file.
        """
        os.remove(os.path.join(self.model_dir, self.model_name))

    def check_if_model_exists(self):
        """
        Check if the model file exists.
        :return: True if the model file exists, False otherwise.
        """
        #question what about the file extension?
        #explanation: the model name is the file name, so we need to check if the file exists
        return os.path.exists(os.path.join(self.model_dir, self.model_name))