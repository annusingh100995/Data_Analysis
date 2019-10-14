""" I attempted to learn the racing bar graph famous on internet
    I followed a blog and learn to drwa the racing graphs using flourish
    So, basically you have to arrange the  data in a particular format and
    the rest is taken care by Flourish software
    I followed this https://towardsdatascience.com/step-by-step-tutorial-create-a-bar-chart-race-animation-da7d5fcd7079"""

# Loading the data 
# I am loadig the data from Cyprus’ Open Data Portal. 
# More precisely the dataset contains the number of people that applied for 
# an unemployment allowance per economic activity category each month since 2013.

import pandas as pd
import re

data_url = "https://raw.githubusercontent.com/apogiatzis/race-bar-chart-unemployment/master/data/unemployment_per_economic_activity_monthly.csv"
df = pd.read_csv(data_url)
df.head()

""" ['NACE 2 CODE', 'Activity Code', 'Economic Activity', 'Year', 'Month',
       'Anastoles-Metapoiisi', 'Anastoles-Touristiki',
       'Termatismoi-Touristiki', 'Termatismoi-Alloi', 'Total'],
      dtype='object') 
      These are the column names """

# SO here the data is in cypurs language and we'll convert it in english 

# Translating the month
# df.Month.unique(), can be used to get the months
month_gr2en = {'Ιανουάριος': '_01 Jan', 'Φεβρουάριος': '_02 Feb', 'Μάρτιος': '_03 Mar',
               'Απρίλιος': '_04 Apr', 'Μάιος':'_05 May', 'Ιούνιος':'_06 Jun',
               'Ιούλιος': '_07 Jul', 'Αύγουστος':'_08 Aug', 'Σεπτέμβριος': '_09 Sep',
               'Οκτώβριος':'_10 Oct', 'Νοέμβριος':'_11 Nov',
               'Δεκέμβριος': '_12 Dec'}


# group byy economic activity
df_group_by_economic_activity = df.groupby('Economic Activity')

# group by year and month 
df_group_by_year_and_month = df.groupby(['Year', 'Month'])

# df.groupby(['Year', 'Month']).groups.keys() will give the indexes of the groups, here it will give the year and months

EA_TOTAL_COLUMNS = [2, 9] # Indices of Economic Activity and Total columns

# this creates an empty df with the keys provided by the following script as the column names
formatted_df = pd.DataFrame(index=df_grouped_by_economic_activity.groups.keys()) # Empty DF, only indices

