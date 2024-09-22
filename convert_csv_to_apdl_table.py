import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

# PyQt5 setup to open file dialog for selecting the CSV file
class FileSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.init_ui()

    def init_ui(self):
        # Open file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            self.file_path = file_path
        self.close()

def select_file():
    app = QApplication(sys.argv)
    selector = FileSelector()
    app.exec_()  # Start the application event loop
    return selector.file_path

# Function to process the CSV file and convert it to a 2D array
def process_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check for at least three columns and the 'cnof' column
    if df.shape[1] < 3 or 'cnof' not in df.columns:
        raise ValueError("CSV file must contain at least two coordinate columns and a 'cnof' column")

    # Assume the first two columns are the coordinates and the last column is 'cnof'
    coord_columns = df.columns[:2]
    value_column = 'cnof'

    # Get unique values for each coordinate to create the axes of the 2D array
    coord1_unique = np.sort(df[coord_columns[0]].unique())
    coord2_unique = np.sort(df[coord_columns[1]].unique())

    # Initialize a 2D array with NaN values to hold the cnof values
    array = np.full((len(coord1_unique), len(coord2_unique)), np.nan)

    # Fill the 2D array using the values from the DataFrame
    for _, row in df.iterrows():
        coord1_idx = np.where(coord1_unique == row[coord_columns[0]])[0][0]
        coord2_idx = np.where(coord2_unique == row[coord_columns[1]])[0][0]
        array[coord1_idx, coord2_idx] = row[value_column]

    # Convert the array to a DataFrame for better visualization (optional)
    result_df = pd.DataFrame(array, index=coord1_unique, columns=coord2_unique)
    return result_df

# Main function to select file, process it and display the result
def main():
    # Step 1: Open the file dialog to select the CSV file
    file_path = select_file()
    
    # Step 2: Process the selected CSV file
    if file_path:
        result_df = process_csv(file_path)
		

if __name__ == '__main__':
    main()
