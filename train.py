import os
import argparse
import joblib

import pandas as pd
import numpy as np
from scipy.sparse import data
from azureml.core.run import Run
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.data.dataset_factory import TabularDatasetFactory

dataset_url = "https://raw.githubusercontent.com/gaganmanku96/Capstone-Project-Azure-Machine-Learning/master/students_placement_data.csv.csv"

ds = TabularDatasetFactory.from_delimited_files(dataset_url)
# ds = pd.read_csv('students_placement_data.csv')

# ds = ds.to_pandas_dataframe()

def get_data(dataframe):
    dataframe.drop('roll_no', axis=1, inplace=True)
    male_female_mapping = {'M': 0, 'F': 1}
    section_mapping = {'A': 0, 'B': 1}
    placement_registeration_mapping = {'NO': 0, 'YES': 1}
    placed_notplaced_mapping = {'Not placed': 0, 'Placed': 1}
    dataframe['gender'] = dataframe['gender'].map(male_female_mapping)
    dataframe['section'] = dataframe['section'].map(section_mapping)
    dataframe['registered_for_placement_training'] = dataframe['registered_for_placement_training'].map(placement_registeration_mapping)
    dataframe['placement_status'] = dataframe['placement_status'].replace(placed_notplaced_mapping)

    dataframe = dataframe.fillna(dataframe.median())

    y = dataframe.pop('placement_status')
    x = dataframe
    return x, y
   
x, y = get_data(ds)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=4)

run = Run.get_context()
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_est', type=int, default=50, help="Number of Estimators")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum samples split")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = RandomForestClassifier(n_estimators=args.n_est,min_samples_split=args.min_samples_split).fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
