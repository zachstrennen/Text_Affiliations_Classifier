import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sklearn.naive_bayes
import jaro

def read_data(path:str) -> pd.DataFrame:
    """
    Read in the dataset from a selected directory.
    Adjust dataframe to specific format.
    :param path: String containing the path name.
    :return: Adjusted dataset pulled from directory path.
    """
    # Read in data from path
    df = pd.read_csv(path)

    # Use the following line to shrink the data for testing purposes
    #df = df.iloc[0:20,]

    # Convert names to a numpy
    affil_array = df.iloc[:,0].to_numpy()

    # Get length of list
    size = len(df.index)

    # Create a data frame that has all possible combinations without repetition
    # (Keep track of indices)
    out=[]
    j=0
    for i in range(0,size):
        for j in range(i+1, size):
            out.append((affil_array[i], affil_array[j], i, j))
    new_df = pd.DataFrame(out, columns=['aff1','aff2','index1','index2'])

    return new_df


def split_data(df:pd.DataFrame,ratio:float,target:str) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    """
    Split a dataframe by target and predictor variables.
    Decide which data will be test data and which data will be training data.
    :param df: Dataframe to be split.
    :param ratio: Ratio (float) of split between training and testing.
    :param target: String of target column name.
    :return: 4 dataframes - the train data for X and y, the test data for X and y.
    """
    X = df[df.columns[~df.columns.isin([target])]]
    y = df[[target]]
    train_X,test_X, train_y,test_y = train_test_split(X,y,test_size=ratio)
    return train_X,test_X, train_y,test_y

def train_data_XGB(train_X:pd.DataFrame,train_y:pd.DataFrame):
    """
    Build the model using XGBoost.
    Print out the accuracy of the model.
    Return the model.
    :param train_X: Train data for X.
    :param train_y: Train data for y.
    :return: Prediction model.
    """
    model = XGBClassifier()
    model.fit(train_X, train_y)
    return model

def train_data_bayes(train_X:pd.DataFrame,train_y:pd.DataFrame):
    """
    Build the model using Naive Bayes.
    Print out the accuracy of the model.
    Return the model.
    :param train_X: Train data for X.
    :param train_y: Train data for y.
    :return: Prediction model.
    """
    model = sklearn.naive_bayes.GaussianNB()
    model.fit(train_X, train_y.values.ravel(),sample_weight=None)
    return model

def possess(model,test_X:pd.DataFrame,test_y:pd.DataFrame):
    """
    Take in the fitted model and use test data to output the models accuracy.
    :param model: Already built model.
    :param test_X: train data for X.
    :param test_y: train data for y.
    :return:
    """
    y_pred = model.predict(test_X)
    predictions = [round(value) for value in y_pred]
    # Compare for accuracy
    accuracy = accuracy_score(test_y, predictions)
    return accuracy

def predict(model, df:pd.DataFrame):
    """
    Take in the model and a dataset of value to use for prediction.
    Use the model on the dataset and return a binary vector of predictions.
    :param model: Already built model.
    :param df: Dataframe specific to the models features.
    :return: Binary vector of predictions.
    """
    # Fit the model to the dataset that will be used for prediction
    predictions = model.predict(df)
    predictions = [round(value) for value in predictions]
    # Return a binary vector of predictions
    return predictions


if __name__ == '__main__':

    # Read in the main data as all possible compinations with respect to indices
    df = read_data("/Users/zachstrennen/Downloads/affiliations.csv")

    # Manually generated strings; Training data for jaro-winkler
    training_data = [['lnfm P7n9, kLq,87bNp5a', 'lnfm P7n9, kLq,87bNp5a',1],
                     ['TaL -ARWy', 'j.ulia wajijnkf', 0],
                     ['ml4ku0X19QgVD9', 'ml4ku0X', 0],
                     ['f62r69Am,f9YJ', 'f62r69Amf9YJ', 1],
                     ['rR2u1mRDC6G', 'fmP7n9kLq', 0],
                     ['zqH7BK40lE7zUYfq','zqH7BK40lE7zUYfq',1],
                     ['1vT2dZHH6HSxFGn','1vT2dZHH6HSxFG',1],
                     ['1vT2dZHH6HSxFGn', '1v3fdZHH6HSxFG', 0],
                     ['6b2YK','dZHH6',0],
                     ['6b2YK','6b2KI',0]]

    # Convert the data above into a data frame
    training_df_setup = pd.DataFrame(training_data, columns=['aff1', 'aff2','match'])

    # Create a vector of scores generated from jaro-winker for each combination
    jw_score_list = []
    for i in range(0,len(training_df_setup)):
        jw_score_list.append(jaro.jaro_winkler_metric(training_df_setup['aff1'][i], training_df_setup['aff2'][i]))

    # Create a data frame that links the scores to the binary match vector from the original training data
    match_list = training_df_setup['match'].values
    training_df = pd.DataFrame({'jw_match': jw_score_list,
                                    'match': match_list})

    # Use all of the data to train the model
    train_X, test_X, train_y, test_y = split_data(training_df, 1,'match')
    model = train_data_XGB(train_X,train_y)

    # Generate a list of jw scores for all combinations in the main data
    jw_score_main = []
    for i in range(0,len(df.index)):
        jw_score_main.append(jaro.jaro_winkler_metric(df['aff1'][i], df['aff2'][i]))

    # Convert the list to a data frame in order for the prediction to occur
    prediction_df = pd.DataFrame({'jw_score': jw_score_main})

    # Get the prediction vector for what may be a considered a match
    df['match'] = predict(model,prediction_df)

    # Filter out what does not match
    df_match = df.loc[df['match'] == 1]

    # Only show the indices that match and store them in a data frame
    df_indices = df_match.iloc[0:len(df_match),2:4]
    print(df_indices)




