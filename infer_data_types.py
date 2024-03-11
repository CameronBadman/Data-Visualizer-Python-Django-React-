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
            non_numeric_count = numeric_series.isna().sum()
            num_values = len(self._file[column])
            threshold = logarithmic_threshold(num_values)
            non_numeric_percentage = non_numeric_count / num_values
            if non_numeric_percentage <= threshold:
                return 'numeric'
        except TypeError:
            return False

    def infer_datetime(self, data):
        if data.dtype == 'datetime64[ns]':
            return 'datetime64[ns]'
        elif data.dtype == 'timedelta64[ns]':
            return 'timedelta64[ns]'
        
        # Check if column contains boolean values
        if data.dtype == bool:
            return False

        # Attempt general datetime conversion without specifying format first
        general_conversion = pd.to_datetime(data, errors='coerce')
        if not general_conversion.isnull().all():
            return 'datetime64[ns]'
        
        # Define possible datetime formats to check
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in formats:
            converted_data = pd.to_datetime(data, format=fmt, errors='coerce')
            if not converted_data.isnull().all():
                return fmt
        
        # If none of the above checks succeed, attempt to convert data to timedelta
        converted_data = pd.to_timedelta(data, errors='coerce')
        if not converted_data.isnull().all():
            return 'timedelta64[ns]'

        return False

    def infer_data_type(self):
        type_inferences = {}
        for column in self._file.columns:
            numeric_type = self.infer_numeric(column)
            datetime_type = self.infer_datetime(self._file[column])
            type_inferences[column] = {
                'datetime': datetime_type if datetime_type else False,
                'numeric': numeric_type if numeric_type else False,
                'categorical': self.infer_categorical(self._file[column]),
                'boolean': self.infer_boolean(self._file[column])
            }
        return type_inferences

    def infer_boolean(self, data):
        unique_values = data.dropna().unique()
        boolean_values = {True, False, 'yes', 'no', 'Yes', 'No', 'TRUE', 'FALSE'}
        if len(unique_values) == 2 and set([str(val).lower() for val in unique_values]).issubset(set([str(val).lower() for val in boolean_values])):
            return True
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
            elif types['datetime'] and types['datetime'] != 'False':
                self._file[column] = pd.to_datetime(self._file[column], errors='coerce')
            elif types['datetime'] == 'timedelta64[ns]':
                self._file[column] = pd.to_timedelta(self._file[column], errors='coerce')
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
    max_column_length = max(len(column) for column in data_types) + 3  # Space for column names
    max_attr_length = max(len(attr) for attr in all_attributes) + 3  # Space for attribute names
    
    # Print the header
    header = f"{'Column Name'.ljust(max_column_length)}| " + " | ".join(attr.ljust(max_attr_length) for attr in sorted(all_attributes))
    print(header)
    print("-" * len(header))
    
    # Print each column's data types
    for column, attributes in data_types.items():
        row = [column.ljust(max_column_length)] + [str(attributes.get(attr, 'N/A')).ljust(max_attr_length) for attr in sorted(all_attributes)]
        print(" | ".join(row))

        
#df = pd.read_csv('sample_data.csv')
test = Frame_Driver('test_data_10000.csv')
print(test.get_DataFrame())
print_dynamic_data_types(test.get_type_inferences())
print("Data types before inference:")

