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
# df.Month.unique(), can be used to get the months _0x is added so that it will be convinient to sort these later
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
formatted_df = pd.DataFrame(index=df_group_by_economic_activity.groups.keys()) # Empty DF, only indices

""" df_group_by_year_and_month.groups.items()return a dictionnary with the groups as keys and the rest as items. 
    (2017, 'Φεβρουάριος') Int64Index([1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186,
            1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197,
            1198, 1199],
           dtype='int64')
           (2017, 'Φεβρουάριος') is the key and k[0] is the first element of the key that is 2017
"""
"""the keys are the year and month and the values are  the number of people that applied for 
# an unemployment allowance per economic activity category each month for the particular year 
# df.iloc[v] will give the df(original) data + the columns [2,9] that is indexed using the economic acticity 
# """
for k,v in df_group_by_year_and_month.groups.items():
    #print(k,'Value is : ',v)
    #print(k[0])
    # we want thr columnlabel to be in "2017 January" format.
    column_label = str(k[0]) + ' ' + month_gr2en[k[1]]
    #print(df.iloc[v])
    #test = df.iloc[v, EA_TOTAL_COLUMNS]
    #print(df.iloc[[1,2],[2,9]])
    # So here, we take the original column, then select the row with the entry correspoing to the keys of the dictionary
    # and then also take the columns 2 and 9 corresponding the economil activitya and the total
    # this is then finally indexed wrt the economic activity
    aggregated = df.iloc[v, EA_TOTAL_COLUMNS].set_index('Economic Activity')
    #print(aggregated)
    aggregated.rename(columns={'Total': column_label}, inplace=True)
    # here the heaading of the column "total" is replaced be the column_labels
    #print(aggregated)
    formatted_df = pd.concat([formatted_df, aggregated], axis=1, join_axes=[formatted_df.index])  

"""formatted_df.columns
has these many columns and econimic activities as the index
Index(['2013 _04 Apr', '2013 _08 Aug', '2013 _12 Dec', '2013 _01 Jan',
       '2013 _07 Jul', '2013 _06 Jun', '2013 _05 May', '2013 _03 Mar',
       '2013 _11 Nov', '2013 _10 Oct', '2013 _09 Sep', '2013 _02 Feb',
       '2014 _04 Apr', '2014 _08 Aug', '2014 _12 Dec', '2014 _01 Jan',
       '2014 _07 Jul', '2014 _06 Jun', '2014 _05 May', '2014 _03 Mar',
       '2014 _11 Nov', '2014 _10 Oct', '2014 _09 Sep', '2014 _02 Feb',
       '2015 _04 Apr', '2015 _08 Aug', '2015 _12 Dec', '2015 _01 Jan',
       '2015 _07 Jul', '2015 _06 Jun', '2015 _05 May', '2015 _03 Mar',
       '2015 _11 Nov', '2015 _10 Oct', '2015 _09 Sep', '2015 _02 Feb',
       '2016 _04 Apr', '2016 _08 Aug', '2016 _12 Dec', '2016 _01 Jan',
       '2016 _07 Jul', '2016 _06 Jun', '2016 _05 May', '2016 _03 Mar',
       '2016 _11 Nov', '2016 _10 Oct', '2016 _09 Sep', '2016 _02 Feb',
       '2017 _04 Apr', '2017 _08 Aug', '2017 _12 Dec', '2017 _01 Jan',
       '2017 _07 Jul', '2017 _06 Jun', '2017 _05 May', '2017 _03 Mar',
       '2017 _11 Nov', '2017 _10 Oct', '2017 _09 Sep', '2017 _02 Feb'],
      dtype='object')"""

# to sort the columns in order of the month
sorted_columns = list(formatted_df.columns.sort_values())

## Add category field + the sorted columns
formatted_df['category'] = range(len(df_group_by_economic_activity.groups.keys()))
formatted_df = formatted_df[['category'] + sorted_columns]

## Remove the prefix used for ordering from the month oclumns
formatted_df.columns = list(map(lambda col: re.sub(r"_[0-9]*[ \t]+","", col),
                               formatted_df.columns))

## Set Index
formatted_df.index.name = 'Economic Activity'

## Remove Unwanted Eaconomic Activity categories
formatted_df = formatted_df.drop(['Μη δηλωμένη οικονομική δραστηριότητα', 'Σύνολο'])

## Save to csv
formatted_df.to_csv('data.csv', encoding='utf-8-sig')
df_small = formatted_df.iloc[0:5,:]

df_small.to_csv(r'D:\C++\PYTHON\ml\data_small.csv', encoding='utf-8-sig')


formatted_df.head()

# https://public.flourish.studio/visualisation/776199/
# SMALL DATA 
# https://public.flourish.studio/visualisation/776296/