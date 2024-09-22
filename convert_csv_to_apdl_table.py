import pandas as pd
import numpy as np
import sys

# Function to process the CSV file and convert it to a 2D array
def process_csv(file_path):
    df = pd.read_csv(file_path)
    if df.shape[1] < 3 or 'cnof' not in df.columns:
        raise ValueError("CSV file must contain at least two coordinate columns and a 'cnof' column")
    coord_columns = df.columns[:2]
    value_column = 'cnof'
    coord1_unique = np.sort(df[coord_columns[0]].unique())
    coord2_unique = np.sort(df[coord_columns[1]].unique())
    array = np.full((len(coord1_unique), len(coord2_unique)), np.nan)
    for _, row in df.iterrows():
        coord1_idx = np.where(coord1_unique == row[coord_columns[0]])[0][0]
        coord2_idx = np.where(coord2_unique == row[coord_columns[1]])[0][0]
        array[coord1_idx, coord2_idx] = row[value_column]
    result_df = pd.DataFrame(array, index=coord1_unique, columns=coord2_unique)
    result_df.index.name = coord_columns[0]
    result_df.columns.name = coord_columns[1]
    return result_df

# Function to create APDL table
def create_apdl_table(result_df, table_name="my_table"):
    # APDL header
    row_index_name = result_df.index.name
    col_index_name = result_df.columns.name
    num_rows, num_cols = result_df.shape
    apdl_lines = []
    
    # DIM command
    apdl_lines.append(f"*DIM,{table_name},{num_rows},{num_cols},1,{row_index_name},{col_index_name}\n")
    
    # Add row index values
    for i, row_index_value in enumerate(result_df.index, start=1):
        apdl_lines.append(f"*SET,{table_name}({i},0,1),{row_index_value}\n")
    
    # Add column index values
    for j, col_index_value in enumerate(result_df.columns, start=1):
        apdl_lines.append(f"*SET,{table_name}(0,{j},1),{col_index_value}\n")
    
    # Add table values
    for i, row_index in enumerate(result_df.index, start=1):
        for j, col_index in enumerate(result_df.columns, start=1):
            data_value = result_df.loc[row_index, col_index]
            if pd.notna(data_value):  # Only write non-NaN values
                apdl_lines.append(f"*SET,{table_name}({i},{j},1),{data_value}\n")
    
    return apdl_lines

# Main function to process the file and display the result
def main(file_path, table_name="my_table"):
    # Step 1: Process the selected CSV file
    if file_path:
        result_df = process_csv(file_path)
        
        # Step 2: Generate APDL table
        apdl_lines = create_apdl_table(result_df, table_name)
        
        # Step 3: Output the APDL table to a file or print
        apdl_file = f"{table_name}.txt"
        with open(apdl_file, 'w') as f:
            f.writelines(apdl_lines)
        print(f"APDL table written to {apdl_file}")
    else:
        print("No file selected")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        main(file_path)
    else:
        print("Please provide the file path as an argument.")
