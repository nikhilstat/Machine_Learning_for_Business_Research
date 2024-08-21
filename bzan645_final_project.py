import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Preprocessing:
    '''This class contains the basic functions used to preprocess the data.
    You should be able to use the functions on any data, 
    not just on the data initiated using the PetFinder class. '''
    
    def remove_outliers(df, column_name ='Age'):
        # For dogs
        dog_q1 = df[(df['Type'] == 1)][column_name].quantile(0.25)
        dog_q3 = df[(df['Type'] == 1)][column_name].quantile(0.75)
        dog_iqr = dog_q3 - dog_q1
        dog_lower_bound = dog_q1 - 1.5 * dog_iqr
        dog_upper_bound = dog_q3 + 1.5 * dog_iqr

        # For cats
        cat_q1 = df[(df['Type'] == 2)][column_name].quantile(0.25)
        cat_q3 = df[(df['Type'] == 2)][column_name].quantile(0.75)
        cat_iqr = cat_q3 - cat_q1
        cat_lower_bound = cat_q1 - 1.5 * cat_iqr
        cat_upper_bound = cat_q3 + 1.5 * cat_iqr

        # Filter out the outliers
        df = df[((df['Type'] == 1) & (df[column_name] >= dog_lower_bound) & 
                 (df[column_name]<= dog_upper_bound)) |
                ((df['Type'] == 2) & (df[column_name] >= cat_lower_bound) & 
                 (df[column_name] <= cat_upper_bound))]

        return df
    
    
    def preprocess_age_data(data, age_column = 'Age'):
        """
        Preprocess the age data by removing negative values, applying log transformation,
        and removing outliers.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        age_column (str): The name of the column representing age.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        # Remove observations with negative age
        processed_data = data[data[age_column] > 0]

        # Apply log transformation to the age column
        processed_data[f'Log_{age_column}'] = np.log(processed_data[age_column])

        processed_data = Preprocessing.remove_outliers(processed_data)

        return processed_data
    

    def simplify_categories(df, column_name, top_n):
        """
        Simplify the categories in a DataFrame column by keeping only the top N categories
        and replacing the rest with a common value.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to simplify.
        top_n (int): The number of top categories to keep.

        Returns:
        pd.DataFrame: The DataFrame with the simplified column.
        """
        # Identify the top N categories in the column
        top_categories = df[column_name].value_counts().head(top_n).index

        # Replace categories not in the top N with a common value (e.g., 999)
        df[column_name] = df[column_name].apply(lambda x: x if x in top_categories else 999)

        return df

    
    def categorical_to_continuous(data, column_names):
        """
        Convert multiple categorical variables to continuous variables based on their 
        proportion 
        in
        the dataset.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_names (list of str): The names of the columns to convert to continuous.

        Returns:
        pd.DataFrame: A DataFrame containing the proportion measures for each specified 
        column.
        """
        continuous_columns = pd.DataFrame()

        for column_name in column_names:
            # Calculate the proportion of each category in the specified column
            category_proportions = data[column_name].value_counts() / len(data)

            # Map each row in the original data to its corresponding proportion
            continuous_columns[column_name] = data[column_name].map(category_proportions)

        return continuous_columns


    
    def create_vectors_for_categorical_variables(data, column_names):
        """
        Create vectors for categorical values in specified columns based on adoption speed.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_names (list of str): The names of the columns to use for vector creation.

        Returns:
        dict: A dictionary where each key is a column name and each value is a DataFrame. 
        Each DataFrame
              represents the vector components for the corresponding column.
        """
        vectors_dict = {}

        for column_name in column_names:
            # Apply the existing logic to each column
            grouped_data = pd.DataFrame(data.groupby(by=[column_name, 
                                                         'AdoptionSpeed']).size(), 
                                        columns=['Count']).reset_index()
            total_count = grouped_data.groupby(column_name).Count.sum()
            merged_data = grouped_data.merge(total_count, left_on=column_name, 
                                             right_index=True, 
                                             suffixes=('', '_Total'))
            merged_data['Proportion'] = merged_data['Count'] / merged_data['Count_Total']
            vectors = merged_data.pivot(index=column_name, columns='AdoptionSpeed', 
                                        values='Proportion')
            vectors.columns = [f'AdoptionSpeed_{col}' for col in vectors.columns]
            vectors = vectors.fillna(0)

            vectors_dict[column_name] = vectors

        return vectors_dict
    
    def merge_vectors_to_dataframe(data, vectors_dict):
        """
        Merge the vector representations of categorical variables back into the original DataFrame.

        Parameters:
        data (pd.DataFrame): The original DataFrame.
        vectors_dict (dict): A dictionary of DataFrames representing vectorized categorical variables.

        Returns:
        pd.DataFrame: The DataFrame with vectorized categorical variables merged in.
        """
        for column_name, vectors_df in vectors_dict.items():
            # Merge the vectors DataFrame with the original DataFrame
            data = data.merge(vectors_df, left_on=column_name, right_index=True, how='left')

            # Optionally, drop the original categorical column
            data = data.drop(column_name, axis=1)

        return data








    












