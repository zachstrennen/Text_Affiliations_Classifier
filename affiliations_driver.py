from affiliations_classifier import read_data, split_data, predict, train_data_XGB
import pandas as pd
import jaro

if __name__ == '__main__':

    # Read in the main data as all possible combinations with respect to indices
    df = read_data("data/affiliations.csv")

    # Manually generated strings; Training data for jaro-winkler
    training_data = [['lnfm P7n9, kLq,87bNp5a', 'lnfm P7n9, kLq,87bNp5a', 1],
                     ['TaL -ARWy', 'j.ulia wajijnkf', 0],
                     ['ml4ku0X19QgVD9', 'ml4ku0X', 0],
                     ['f62r69Am,f9YJ', 'f62r69Amf9YJ', 1],
                     ['rR2u1mRDC6G', 'fmP7n9kLq', 0],
                     ['zqH7BK40lE7zUYfq', 'zqH7BK40lE7zUYfq', 1],
                     ['1vT2dZHH6HSxFGn', '1vT2dZHH6HSxFG', 1],
                     ['1vT2dZHH6HSxFGn', '1v3fdZHH6HSxFG', 0],
                     ['6b2YK', 'dZHH6', 0],
                     ['6b2YK', '6b2KI', 0]]

    # Convert the data above into a data frame
    training_df_setup = pd.DataFrame(training_data, columns=['aff1', 'aff2', 'match'])

    # Create a vector of scores generated from jaro-winker for each combination
    jw_score_list = []
    for i in range(0, len(training_df_setup)):
        jw_score_list.append(
            jaro.jaro_winkler_metric(training_df_setup['aff1'][i],
                                     training_df_setup['aff2'][i]
                                     ))

    # Create a data frame that links the scores to the binary match
    # vector from the original training data
    match_list = training_df_setup['match'].values
    training_df = pd.DataFrame({'jw_match': jw_score_list,
                                'match': match_list})

    # Use all of the data to train the model
    train_X, test_X, train_y, test_y = split_data(training_df, 1, 'match')
    model = train_data_XGB(train_X, train_y)

    # Generate a list of jw scores for all combinations in the main data
    jw_score_main = []
    for i in range(0, len(df.index)):
        jw_score_main.append(jaro.jaro_winkler_metric(df['aff1'][i], df['aff2'][i]))

    # Convert the list to a data frame in order for the prediction to occur
    prediction_df = pd.DataFrame({'jw_score': jw_score_main})

    # Get the prediction vector for what may be a considered a match
    df['match'] = predict(model, prediction_df)

    # Filter out what does not match
    df_match = df.loc[df['match'] == 1]

    # Only show the indices that match and store them in a data frame
    df_indices = df_match.iloc[0:len(df_match), 2:4]
    print(df_indices)
