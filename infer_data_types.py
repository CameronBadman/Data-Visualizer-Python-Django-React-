import pandas as pd
import numpy as np

class Frame_Driver:
    def __init__(self, filepath):
        self._file = pd.read_csv(filepath)
        self._type_inferences = self.infer_data_type()
        self.convert_data_types()

    def get_DataFrame(self):
        return self._file
    
    def get_type_inferences(self):
        return self._type_inferences

    def infer_numeric(self, column):
        try:
            numeric_series = pd.to_numeric(self._file[column], downcast='integer', errors='coerce')
            # Count the number of non-numeric values
            non_numeric_count = numeric_series.isna().sum()
            
            # Calculate the percentage of non-numeric values
            num_values = len(self._file[column])
            threshold = logarithmic_threshold(num_values)
            non_numeric_percentage = non_numeric_count / num_values

            # If the percentage of non-numeric values is below the threshold, consider the column numeric
            if non_numeric_percentage <= threshold:
                    return 'numeric'  # Default to 'numeric' for other numeric data types

        except TypeError:
            return False

    def infer_data_type(self):
        type_inferences = {}
        for column in self._file.columns:
            numeric_type = self.infer_numeric(column)
            type_inferences[column] = {
                'datetime': self.infer_datetime(self._file[column]),
                'numeric': numeric_type if isinstance(numeric_type, str) else False,
                'categorical': self.infer_categorical(self._file[column])
            }
        return type_inferences


    def infer_datetime(self, data):
        # Check if the data is already in Datetime64 or Timedelta[ns] format
        print(data.dtype)
        if data.dtype == 'datetime64[ns]':
            return 'datetime64[ns]'
        elif data.dtype == 'timedelta64[ns]':
            return 'timedelta64[ns]'
        
        # Check for string datetime formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in formats:
            try:
                pd.to_datetime(data, format=fmt)
                return fmt
            except (TypeError, ValueError):
                pass
    
    # If the data is not in any datetime format, return False
        return False

    def infer_categorical(self, data):
        if isinstance(data.dtype, pd.CategoricalDtype):
            return True
        elif data.nunique() < len(data) / 1.1:
            return True
        elif pd.api.types.infer_dtype(data, skipna=True) == 'category':
            return True
        else:
            return False

    def convert_data_types(self):
        for column, types in self._type_inferences.items():
            if types['numeric']:
                self._file[column] = pd.to_numeric(self._file[column], downcast='integer', errors='coerce')
            elif types['datetime']:
                self._file[column] = pd.to_datetime(self._file[column], errors='coerce')
            elif types['categorical']:
                self._file[column] = self._file[column].astype('category')

def logarithmic_threshold(num_values, multiplier=3):
    """Compute a logarithmic threshold based on the number of values."""
    return np.log10(num_values) / np.log10(num_values * multiplier)

def print_dynamic_data_types(data_types):
    """
    Prints data types information dynamically for any set of columns and data type attributes.
    
    Parameters:
    data_types (dict): A dictionary with columns as keys and dictionaries of data type attributes as values.
    """
    # Extract unique attributes across all columns for dynamic headers
    all_attributes = set(attr for dt in data_types.values() for attr in dt)
    max_column_length = max(len(column) for column in data_types) + 2  # Space for column names
    max_attr_length = max(len(attr) for attr in all_attributes) + 2  # Space for attribute names
    
    # Print the header
    header = f"{'Column Name'.ljust(max_column_length)}| " + " | ".join(attr.ljust(max_attr_length) for attr in sorted(all_attributes))
    print(header)
    print("-" * len(header))
    
    # Print each column's data types
    for column, attributes in data_types.items():
        row = [column.ljust(max_column_length)] + [str(attributes.get(attr, 'N/A')).ljust(max_attr_length) for attr in sorted(all_attributes)]
        print(" | ".join(row))

        
#df = pd.read_csv('sample_data.csv')
test = Frame_Driver('test_data_extension.csv')
print(test.get_DataFrame())
print_dynamic_data_types(test.get_type_inferences())
print("Data types before inference:")

