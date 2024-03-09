# %%

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
            # Attempt to coerce values to numeric
            numeric_series = pd.to_numeric(self._file[column], errors='coerce')
            # Count the number of non-numeric values
            non_numeric_count = numeric_series.isna().sum()
            
            # Calculate the percentage of non-numeric values
            num_values = len(self._file[column])
            threshold = logarithmic_threshold(num_values)
            non_numeric_percentage = non_numeric_count / num_values

            # If the percentage of non-numeric values is below the threshold, consider the column numeric
            return non_numeric_percentage <= threshold
        except TypeError:
            return False

    def infer_data_type(self):
        type_inferences = {}
        for column in self._file.columns:
            type_inferences[column] = {
                'datetime': self.infer_datetime(self._file[column]),
                'numeric': self.infer_numeric(column),
                'categorical': self.infer_categorical(self._file[column])
            }
        return type_inferences

    def infer_datetime(self, data):
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in formats:
            try:
                pd.to_datetime(data, format=fmt)
                return True
            except (TypeError, ValueError):
                pass
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
                self._file[column] = pd.to_numeric(self._file[column], errors='coerce')
            elif types['datetime']:
                self._file[column] = pd.to_datetime(self._file[column], errors='coerce')
            elif types['categorical']:
                self._file[column] = self._file[column].astype('category')

def logarithmic_threshold(num_values, multiplier=3):
    """Compute a logarithmic threshold based on the number of values."""
    return np.log10(num_values) / np.log10(num_values * multiplier)

# Test the function with your DataFrame
#df = pd.read_csv('sample_data.csv')
test = Frame_Driver('sample_data.csv')
print(test.get_DataFrame())
print("Data types before inference:")
#data_converted = infer_and_convert_data_types(df)
#print(df.dtypes)
