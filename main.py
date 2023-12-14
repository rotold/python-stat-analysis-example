"""An example of data analysis using a synthetic dataset from kaggle to explore effects of gender
on purchases. Source data are found at
https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis/

Because others published some analysis on this data set, I shall try to do analysis that
is not done by them.  This analysis notes that an individual is not represented by each row
in the source data, but rather by every row containing the 'Customer ID' of the individual
"""

import analyse_data as a
from os.path import exists


datafile_path = 'ecommerce_customer_data_large.csv'
datafile_new_path = 'ecommerce_customer_data_large1.csv'
if not exists(datafile_new_path):
    df = a.clean_data(datafile_new_path, datafile_path)
else:
    print(f"The path {datafile_new_path} already exists; skipping cleaning of data")
    df = a.clean_data(datafile_new_path)

a.tableau_verify(df)

individ_data = a.create_individ_stats(df)
individ_data.to_csv("individual data DataFrame")

# df is no longer needed
del df

# look at categorical data
a.conting_table(individ_data)
# look at non-categorical rate data. Does not mutate the DataFrame Sent to it:
a.rate_tests(individ_data)
