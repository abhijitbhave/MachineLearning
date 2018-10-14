import pandas as pd
import numpy as np

# In the data set available at UCI there are two data sets available, one for red wine and the other for White. I am making
# sure that one variable can be set to select the dataset based on user preference.
data_to_use = 'Red'

#Reading the correct data set using Pandas library

if(data_to_use == 'Red'):
    wine_data = pd.read_csv('winequality-red.csv', sep =';')
else:
    wine_data = pd.read_csv('winequality-white.csv', sep=';')

#Extracting the list of columns from the CSV. CConsidering the first rows as columns, Since this list of columns will be leveraged
#for further processing and normalization of data I have not included the Quality columns in this list since we dont want to normalize
#that column.

columns_list = list(wine_data.columns)[0:-1]

#Function for normalizing data. Each column is considered seperately and normalized dependent on its value compared to the mean and
#standard deviation of the column.

def normalize_data(data, columns):
    for column in columns:
        mean_column_value = data.loc[:,column].mean()
        standard_column_deviation = np.std(data.loc[:,column].values)
        data.loc[:,column] = (data.loc[:,column] - mean_column_value) / standard_column_deviation
    return data

#Calling function to normalize data.
normalized_data = normalize_data(wine_data, columns_list)

#Threshhold value to be used in the outliers finction below will determine the threshold of considering a data value an outlier to be fixed

threshold = 3

#A function to determine and readjust outliers in data values based on a combination of threshold,  mean and standard deviation fo a column
#compared to the data value instance of the column.

def remove_outliers(data, threshold, columns):
    for column in columns:
        column_needs_change = data[column] > float(threshold)*data[column].std()+data[column].mean()
        data.loc[column_needs_change == True, column] = np.nan
        mean_update = data.loc[:,column].mean()
        data.loc[column_needs_change == True, column] = mean_update
    return data

#Calling the function to fix outliers in the data. Note that the input set to this fucntion is the normalized data.
wine_data_with_no_outliers = remove_outliers(normalized_data, threshold, columns_list)



#creating a new copy of the data in memory so that we can add a new column to store the label of the wine as determined by the quality index
data_copy = wine_data_with_no_outliers.copy()

#setting up bins to label wines based on quality index
wine_quality_bins = [3, 5, 8]

#Assigning quality labels to wines based on quality index. Anything that over 7 is Good, anything under is Poor.
data_copy['quality_label'] = pd.cut(data_copy.quality, wine_quality_bins, labels=['Poor', 'Good'], include_lowest = True)


#Saving the data which has Wine Quality Labels, Normalized data and no outliers into a new CSV file that can be used by TensorFlow.
data_copy.to_csv('winedata_pre_processed.csv', index = False)

