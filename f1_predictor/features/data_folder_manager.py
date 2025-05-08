import os

class DataFolderManager():
    "This is a class to manage the data folder for the F1 Predictor project."
    "It checks if the folder exists, creates it if it doesn't, and manages its contents."
    "It also checks if the folder is empty and can empty the folder if needed."
    def __init__(self, empty_folder=False):
        currentdir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(currentdir)
        self.data_folder = os.path.join(parentdir, 'data')
        self.check_data_folder(self.data_folder)
        self.is_folder_empty = self.check_if_folder_empty(self.data_folder)
        if not self.is_folder_empty and empty_folder:
            self.empty_data_folder()

    def check_data_folder(self, folder_path):
        """
        Check if the data folder exists and create it if it doesn't.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Data folder created at: {folder_path}")

    def check_if_folder_empty(self, folder_path):
        """
        Check if the folder is empty.
        """
        if not os.listdir(folder_path):
            #print(f"The folder {folder_path} is empty.")
            return True
        else:
            #print(f"The folder {folder_path} is not empty.")
            return False
        
    def empty_data_folder(self):
        """
        Empty the folder.
        """
        folder_path = self.data_folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    #print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
                    #print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        #print(f"All files in {folder_path} have been deleted.")

    def list_files_in_folder(self):
        """
        List all files in the folder.
        """
        folder_path = self.data_folder
        files = os.listdir(folder_path)
        print(f"Files in {folder_path}: {files}")
        return files
    
    def save_features(self, train_data, test_data):
        train_data.to_csv(os.path.join(self.data_folder ,"train_data.csv"), index=False)
        test_data.to_csv(os.path.join(self.data_folder ,"test_data.csv"), index=False)
        print("CSV files saved: data/train_data.csv and data/test_data.csv")

    def verify_csv_files(self):
        """
        Verify if the CSV files exist in the data folder.
        """
        train_file = os.path.join(self.data_folder, "train_data.csv")
        test_file = os.path.join(self.data_folder, "test_data.csv")
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("CSV files exist.")
            return True
        else:
            print("CSV files do not exist.")
            return False