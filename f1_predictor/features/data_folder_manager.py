import os

class DataFolderManager():
    """
    This is a class to manage the creation, verification and content of the data folder for the F1 Predictor project.
    It checks if the folder exists, creates it if it doesn't, and manages its contents.
    
    This includes checking existence, creating the folder if missing, checking if it's empty,
    clearing its contents, listing stored files, and saving/verifying training and test data.
    """
    def __init__(self, empty_folder=False) -> None:
        """Constructor method to initialize the DataFolderManager and prepare the data directory.

        :param empty_folder: Whether to empty the folder if it already exists.
        :type empty_folder: bool, optional
        """    
        currentdir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(currentdir)
        self.data_folder = os.path.join(parentdir, 'data')
        self.check_data_folder(self.data_folder)
        self.is_folder_empty = self.check_if_folder_empty(self.data_folder)
        if not self.is_folder_empty and empty_folder:
            self.empty_data_folder()

    def check_data_folder(self, folder_path) -> None:
        """
        Check if the data folder exists and create it if it doesn't.

        :param folder_path: Path to the data folder.
        :type folder_path: str
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Data folder created at: {folder_path}")

    def check_if_folder_empty(self, folder_path) -> None:
        """Checks if the folder is empty.

        :param folder_path: _description_
        :type folder_path: _type_
        :return: _description_
        :rtype: _type_
        """
        if not os.listdir(folder_path):
            #print(f"The folder {folder_path} is empty.")
            return True
        else:
            #print(f"The folder {folder_path} is not empty.")
            return False
        
    def empty_data_folder(self) -> None:
        """
        Empty the data folder by removing all files and subdirectories.
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

    def list_files_in_folder(self) -> List[str]:
        """        
        List all files currently present in the data folder.

        :return: List of filenames in the data folder.
        :rtype: List[str]
        """
        folder_path = self.data_folder
        files = os.listdir(folder_path)
        print(f"Files in {folder_path}: {files}")
        return files
    
    def save_features(self, train_data, test_data) -> None:
        """
        Save train and test features to CSV files in the data folder.

        :param train_data: DataFrame containing training data.
        :type train_data: pd.DataFrame
        :param test_data: DataFrame containing test data.
        :type test_data: pd.DataFrame
        """    
        train_data.to_csv(os.path.join(self.data_folder ,"train_data.csv"), index=False)
        test_data.to_csv(os.path.join(self.data_folder ,"test_data.csv"), index=False)
        print("CSV files saved: data/train_data.csv and data/test_data.csv")

    def verify_csv_files(self):
        """
        Check whether both the train and test CSV files exist in the data folder.

        :return: True if both files exist, False otherwise.
        :rtype: bool
        """
        train_file = os.path.join(self.data_folder, "train_data.csv")
        test_file = os.path.join(self.data_folder, "test_data.csv")
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("CSV files exist.")
            return True
        else:
            print("CSV files do not exist.")
            return False