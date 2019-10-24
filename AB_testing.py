# https://blog.dominodatalab.com/ab-testing-with-hierarchical-models-in-python/

import math as mt
import numpy as np 
import pandas as pd 
from scipy.stats import norm  

# defininf the baseline estimators
baseline = {"Cookies":40000, "Clicks":3200, "Enrollments":660,
    "CTP":0.08, "GConversion":0.20625, "Retention":0.53,
    "NConversion":0.109313}

baseline["Cookies"] = 5000
baseline["Clicks"] = baseline["Clicks"]*(5000/40000)
baseline["Enrollments"] = baseline["Enrollments"]*(5000/40000)

# GC : Gross Conversion 
""" The baseline probability for Gross Conversion can be calculated by the number of users to enroll in a free trial divided by the number of cookies clicking the free trial. 
In other words, the probability of enrollment given a click."""
# Defining an empty dictionary
GC = {}
GC["d_min"] =0.01
# the probability of enrollment given a click 
GC["p"] = baseline["GConversion"]
# n is the sample size
GC["n"] = baseline["Clicks"]
# standard deviation 
GC["sd"] = round(mt.sqrt((GC["p"]*(1-GC["p"]))/GC["n"]),4)


"""Retention - The baseline probability for retention is the number of paying users (enrolled after 14 free days) divided by the number of total enrolled users. In other words, the probability of payment, given enrollment. 
The sample size is the number of enrolled users."""

R = {}
R["d_min"] = 0.01
R["p"] = baseline["Retention"]
R["n"] = baseline["Enrollments"]
R["sd"] = round(mt.sqrt((R["p"]*(1-R["p"]))/R["n"]),4)
R["sd"]

"""Net Conversion - The baseline probability for the net conversion is the number of paying users divided by the number of cookies that clicked the free trial button. 
In other words, the probability of payment, given a click."""

NC = {}
NC["d_min"] = 0.0075
NC["p"] = baseline["NConversion"]
NC["n"] = baseline["Clicks"]
NC["sd"] = round(mt.sqrt((NC["p"]*(1-NC["p"]))/NC["n"]),4)

# Defining methods to calculate the sample space

def get_sds(p,d):
    sd1 = mt.sqrt(2*p*(1-p))
    sd2 = mt.sqrt(p*(1-p)+(p+d)+(1-(p+d)))
    x = [sd1, sd2]
    return x 
 
 """ Significance and  Power
    The power of a binary hypothesis test is the probability that the test rejects the null hypothesis (H0) 
    when a specific alternative hypothesis (H1) is true. The statistical power ranges from 0 to 1, and as statistical power increases, 
    the probability of making a type II error (wrongly failing to reject the null hypothesis) decreases.
    
    significance level, denoted α, is the probability of the study rejecting the null hypothesis, 
    given that the null hypothesis were assumed to be true;
     """

""" The minimum sample size for control and experiment groups, which
    provides probability of Type 1 Error (alpha) Power 1- (beta)
    detectable effect d and baseline conversion rate p 
     n=(Z1−α2sd1+Z1−βsd2)2d2
    sd1=p(1−p)+p(1−p))1/2
    sd2=p(1−p)+(p+d)(1−(1−(p+d))1/2
    """
# return z score for a given alpha value    
def get_z_score(alpha):
    return norm.ppf(alpha)
# p: is baseline conversion rate 
# d: is minimum detectable change
def get_sds(p,d):
    sd1 = mt.sqrt(2*p*(1-p))
    sd2 = mt.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    sds = [sd1,sd2]
    return sds

# the minimum sample size required per group according to metric denominator

def get_sampSize(sds,alpha, beta, d):
    n = pow((get_z_score(1-alpha/2)*sds[0]+get_z_score(1-beta)*sds[1]),1)/pow(d,2)
    return n

GC["d"] = 0.01
R["d"] = 0.01
NC["d"] = 0.0075

GC["SampSize"] = round(get_sampSize(get_sds(GC["p"],GC["d"]),0.05,0.2,GC["d"]))
GC["SampSize"]

GC["SampSize"]=round(GC["SampSize"]/0.08*2)
GC["SampSize"]

R["SampSize"]=round(get_sampSize(get_sds(R["p"],R["d"]),0.05,0.2,R["d"]))
R["SampSize"]

R["SampSize"]=R["SampSize"]/0.08/0.20625*2
R["SampSize"]

NC["SampSize"]=round(get_sampSize(get_sds(NC["p"],NC["d"]),0.05,0.2,NC["d"]))
NC["SampSize"]

NC["SampSize"]=NC["SampSize"]/0.08*2
NC["SampSize"]

# Loading the data 

control = pd.read_csv(r'D:\C++\PYTHON\ml\ab_test\control_data.csv')
experiment = pd.read_csv(r'D:\C++\PYTHON\ml\ab_test\experiment_data.csv')

"""
control.columns
Index(['Date', 'Pageviews', 'Clicks', 'Enrollments', 'Payments'], dtype='object')"""


# Step - 1 Sanity Check 

"""This is to verify that the experiment was conducted as expected
    and that other factors did not influence the data and also to make 
    sure that the data collected was correct
    
    3 invariant metrics:
    # cookies in ocurse overview page
    # clicks on free trial button 
    # free trail Click-Through-Probability"""


pageview_cont = control['Pageviews'].sum()
pageview_exp = experiment['Pageviews'].sum()
pageview_total = pageview_cont+pageview_exp

print("Number of pageview in control :", pageview_cont)
print("Number of pageview in experiment :", pageview_exp)

""" We want to make sure that the difference in amount is not 
significant and is random 
The number of page view should not change in the experiment
The experiment starts after the pop up appears when a viewer clicks
on the enrol class button
Being assigned to the control group is considered as success an since 
there are two outcomes, the probability of success is 0.5 
And we want the observed probability that the number of samples
in control divided by the total number of samples should not be significanlty 
different than 0.5.  
TO do that the margin of error acceptable at a 95% confidence level 
The confidence interval is derived to tell us in which range an obaerved
p can exist and be acceptable as the same as the expected value
"""

p = 0.5 
aplha = 0.05
p_hat = round(pageview_cont/(pageview_total), 4)
sd = mt.sqrt(p*(1-p)/(pageview_total))
ME = round(get_z_score(1-(aplha/2))*sd,4)
print("The confidence interval is between ", p-ME, " and ", p+ME, ":is", p_hat, "outside the range?")

"""Since the p_hat is inside the confidence interval this invariant metric test is pass"""


"""Sanity test 2
    The clicks on the free trial button. As said above the clicks on the free trial button \\
        should not change and the probabilty should be 0.5"""

# Number df people who clicked the free trail button 
clicks_cont = control['Clicks'].sum()
clicks_exp = experiment['Clicks'].sum()
clicks_total = clicks_cont+clicks_exp

p_hat = round(clicks_cont/clicks_total,4)
sd = mt.sqrt(p*(1-p)/clicks_total)
ME = round(get_z_score(1-(alpha/2))*sd, 4)
print("The confidence interval is between ", p-ME, " and ", p+ME, "; Is ", p_hat, " inside the confidence interval")

""" The probability of clicks is random as it is in the confidence interval.
    Hence the sanity check is pass"""

""" We want to perform the next sanity check that is to cheak the click through probabilities.
    As the click through probability should not change much with the experiment. 
    we expect to see no difference ( CTPexp−CTPcont=0 ), with an acceptable margin of error, dictated by our calculated confidence interval.
    
    We calculate the standard pool error

    """
# The probability that 
ctp_cont = clicks_cont/pageview_cont
ctp_exp = clicks_exp/pageview_exp
d_hat = round(ctp_exp-ctp_cont,4)
p_pooled = clicks_total/pageview_total
sd_pooled = mt.sqrt(p_pooled*(1-p_pooled)*(1/pageview_cont+1/pageview_exp))
ME = round(get_z_score(1-(aplha/2))*sd_pooled,4)
print("The confidence interval is between", 0-ME, " and ", 0+ME, "Is", d_hat, " within this range?")

""" The next step is to look at the changes between the control and expriment group with regards to the evaluation metrics. 
    We want the chnages to be statistically significant here"""

""" The difference of values between both the groups is measured and then the confidence interval for the difference is 
    calculated. The difference should be statistically as well as proactically signficant"""


# Counting the total clicks where enrollment is also non zero

clicks_cont = control["Clicks"].loc[control["Enrollments"].notnull()].sum()
clicks_exp = experiment["Clicks"].loc[control["Enrollments"].notnull()].sum()

# Gross enrollment -  number of enrollments/number of clicks

enrollments_cont = control["Enrollments"].sum()
enrollments_exp = experiment["Enrollments"].sum()

#GC = Gross Conversion

GC_cont = enrollments_cont/clicks_cont
GC_exp = enrollments_exp/clicks_exp
GC_pooled = (enrollments_cont+enrollments_exp)/(clicks_cont+clicks_exp)
GC_sd_pooled = mt.sqrt(GC_pooled*(1-GC_pooled)*(1/clicks_cont+1/clicks_exp))
GC_ME = round(get_z_score(1-aplha/2)*GC_sd_pooled,4)
GC_diff = round(GC_exp-GC_cont,4)
print("The change dur to the experimental is : " , GC_diff*100, "%")
print("Confidence interval : [", GC_diff-GC_ME, ",", GC_diff+GC_ME,"]")

""" the change is -2.06 %, that is there was 2 % decreas in enrollment after the pop up message"""

""" Second Evaluation Metric: Net Conversion 
    the fraction of payers to decrease as well"""

# Net Conversion : number of payments / number of clicks

payments_cont = control["Payments"].sum()
payments_exp = experiment["Payments"].sum()

NC_cont = payments_cont/clicks_cont
NC_exp = payments_exp/clicks_exp

NC_pooled = (payments_cont+payments_exp)/(clicks_cont+clicks_exp)

NC_sd_pooled = mt.sqrt(NC_pooled*(1-NC_pooled)*(1/clicks_cont+1/clicks_exp))

NC_ME = round(get_z_score(1-aplha/2)*NC_sd_pooled,4)
NC_diff = round(NC_exp-NC_cont,4)

print("The change due to the experiment is :", NC_diff*100 ,"%")
print("Confidence Interval : [", NC_diff-NC_ME, ",", NC_diff+NC_ME, "]")

"""Here the change is -0.49 % That is the net conversion decreades by 0.5 %
    This is a very insignificant change"""

""" DOUBLE CHECK WITH SIGN TEST
    In a sign check the trend of the test is observed on a daily basis. The metric value is calculated for each day,
    and the trend is observed daily for a period of time"""

full = control.join(other=experiment, how="inner", lsuffix="_cont", rsuffix="_exp")
""" The pageviews are recorded for more number of days than enrollments.
    So here the entires with no enrollments are removed"""

full = full.loc[full["Enrollments_cont"].notnull()]

# Derive a new column for each metric, to have daily values

# Gross Conversion on each day
x = full['Enrollments_cont']/full['Clicks_cont']
y = full['Enrollments_exp']/full['Clicks_exp']
full['GC'] = np.where(x<y,1,0)

# Net Conversion on each day 
z = full['Payments_cont']/full['Clicks_cont']
w = full['Payments_exp']/full['Clicks_exp']
full['NC'] = np.where(z<w,1,0)

GC_x = full.GC[full["GC"]==1].count()
NC_x = full.NC[full["NC"]==1].count()
n = full.NC.count()

print("No. of cases for GC: " , GC_x, '\n',
    "No. of cases for NC: ", NC_x, '\n',
    "No. of total cases", n)

""" I counted the number of days where the experimental group
    has higher metric. to see if that number is likely to be seen 
    again in a new experiment.  
    Chances that the day will have higher experimental metric = 0.5 (random)
    and the number of days to tell us the probability of this happening according to 
    random chance. (a Binomial distribution )
    p−value  is the probability of observing a test statistic as or more extreme than that observed. If we observed 2 days like that, the p−value for the test is: p−value=P(x<=2). We only need to remember the following:
    P(x<=2)=P(0)+P(1)+P(2).

    """

def get_prob(x,n):
    p=round(mt.factorial(n)/mt.factorial(x)*mt.factorial(n-x)*0.5**x*0.5**(n-x),4) 
    return p

def get_2side_pvalue(x,n):
    p=0
    for i in range(0,x+1):
        p=p+get_prob(i,n)
    return 2*p

print ("GC Change is significant if",get_2side_pvalue(GC_x,n),"is smaller than 0.05")
print ("NC Change is significant if",get_2side_pvalue(NC_x,n),"is smaller than 0.05")

""" So, in AB testing we define an experiment to test how the two version of a same website/page performs. 
    We define the metrics. Evaluation and  Invariant metrics and fix the values of the significance and statitical power we 
    require. Based on these values, we calculate the data required for analysis.(like number of days )
    After data collection we perform the sanity check to make sure that the data collected makes sense and cna be further analysed. 
    Then the evaluation metrics are calculated and the control and the experimental data is compared. 
    
    The statistical significance of the results and their repective confidence interval is used to make necessary conclusions."""