# Linear-Regression-in-Python
In this repository, I will be taking yoou through the step by step process I went through to create a linear regression model in python using football transfers from 1992 to 2019 in the English Premier League

## Introduction
It's been a while since I put up a data related project. Hopefully, this one would finally be completed :grimacing:🤞. I started my transition from data analytics into the world of data science very recently and in a bid to reinforce what I have picked up so far,
I have decided to have my first data science project be a linear regression model that looks at football transfers from 1992 to 2022 in the English Premier League and with the model, I am going to try to predict the transfer fee a club in the English Premier League would be willing to pay if we had certain varaibles available. I would also be interpreting the results of the model to help determine how relaible the model is going to be. 

I am already getting pretty excited for this one so without any further ado, let's get started! 😃

## Exploring the data
The dataset I used for this project can be found [here](https://github.com/ewenme/transfers/blob/master/data/premier-league.csv). Below is a key for the columns in this dataset.

|Header	|Description	|Data Type|
|-------|-------------|---------|
|club_name|	name of club|	text|
|player_name|	name of player	|text|
|position|	position of player	|text|
|club_involved_name|	name of secondary club involved in transfer	|text|
|fee	|raw transfer fee information	|text|
|transfer_movement	|transfer into club or out of club?	|text|
|transfer_period	|transfer window (summer or winter)	|text|
|fee_cleaned	|numeric transformation of fee, in EUR millions	|numeric|
|league_name	|name of league club_name belongs to	|text|
|year	|year of transfer	|text|
|season	|season of transfer (interpolated from year)	|text|
|country	|country of league	|text|

The first thing I noticed while exploring the dataset was that the fee_cleaned column had some NA values. This was because of one of the following reasons:
- a loan transfer with no fees involved
- a transfer made with an undisclosed amount of money
- a player being promoted from the junior ranks into the senior squad


Looking at the columns in the dataset, I realized that wouldn't be using the league_name, season and country columns so I would have to delete them from the dataset during the cleaning process. 

I also noticed that data included transfers into and out of the premier league.
For the purpose of this project, I only wanted to see transfers into the premier league. I also decided that I would be taking out all loan transfers from the dataset. 
Another change I decided I was going to make was to reduce the number of player positions to 4 different positions i.e Goalkeeper, Defender, Midfielder and Attacker. Having a fair idea of what I wanted my data to look like, I started the data cleaning process in Microsoft Excel.

## Cleaning the data
After importing the csv file into excel, I created a copy of the original data and renamed it "cleaned". This is where I carried out the data cleaning process before I created the regression model in python.

First thing I did was to remove all of the "out" entries in the transfer_movement column. I did this to help remove duplicate transfers and to ensure that all transfer fees would be fees spent by premier league clubs only. 
For example, if a player moved from one premier league team to another, there will be two records of the same transfer one as in and the other as out.
Also, if a player had moved from a premier league club to a club in another league, there will be a record of that transfer as well.
By removing all the out transfers, I took away both of these problems. 

When all external transfers had been removed, I changed the club_name column name to transfer_to and the club_involved_name to transfer_from because I made the transfer movement much clearer for me.

Next, I took out all loan deals made. To do this, I looked up the word "loan" in the fee column and deleted all loan transfer records from the data set.
I did this because I ony wanted to focus on permanent transfers premier league clubs had made from 1992 to 2022.

![removing loans](https://github.com/user-attachments/assets/405601eb-5517-40af-afef-99b05cc21a34)

The next thing I did was to take out all records without a transfer value. While exploring the data, I realized that some transfers had been made where the fee was undisclosed to the public the dataset had also included included promotions from the junior team into the senior squad as transfers. The value in the fee_cleaned column for these transfers had been recorded as NA and because I wanted the model to predict the cost of a player if certain variables are presented, I would have no use for NA values. And just like I took out all loan transfer records, I filtered for NA in the fee_cleaned column and removed all records without a transfer value as well.

Then I had to come up with a way to define the different player positions. As mentioned before, there were too many distinct player positions in the original dataset and I believe that every position on a football field can be grouped into 4 main categories, Goalkeeper, Defender, Midfielder and Attacker. Using excel’s unique formula, I was able to draw all the distinct player positions in the original dataset and assign each one of them into my four main categories as shown below.

![new player positions](https://github.com/user-attachments/assets/7bdb7d2c-669b-44b9-8044-6093ffac891d)

After this, I used excel’s xlookup formula to assign my new player positions.

Finally, I took out all of the columns that I would no longer need, renamed and rearranged the columns in the dataset. My final dataset looked something like this

![final dataset](https://github.com/user-attachments/assets/b0bcb82e-edc3-4459-b80e-cb150eac409a)


And now, I could import the data into python to build the linear regression model

## Building the regression model
```
# importing relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
```
```
# importing the cleaned dataset into jupyter notebooks

prem_transfers = pd.read_excel('premier_league_transfers.xlsx',sheet_name='cleaned')

prem_transfers.head()
```
![prem_transfers head()](https://github.com/user-attachments/assets/e5580c50-35e0-48da-8dd3-c15e51f65b57)

After importing the relevant libraries and the cleaned dataset into python, it was time to start building the regression model.

### Creating dummy variables

At this point, I already knew what my dependent and independent variables were going to be. The dependent variable, which is the model I was going to try to predict was the transfer_fee and the independent variables (predictors) were the year,age,new_position and transfer_period.

However, there was a small problem with the dependent variables. There were two columns there that had categorical data. The new_position and the transfer_period varaibles. To deal with this, I had to create dummy varaibles for both these columns since regression models can only work with numerical data. 

What is a dummy varaible? According to wikipedia, a dummy variable is one that takes a binary value (0 or 1) to indicate the absence or presence of some categorical effect. 

For the transfer_period, since there were only two categories so I set Summer as 0 and winter as 1.

However, because there were 4 different categories in the new_position column, we had to create multiple dummy variables. The code is as follows:
```
#creating a copy of the original dataset so I can input the dummy variables
pl_transfers = prem_transfers.copy()

#setting summer transfers as 0 and winter transfers as 1
pl_transfers['transfer_period'] = pl_transfers['transfer_period'].map({'Summer':0,'Winter':1})

#creating dummy variables for new_position
pos_dummies = pd.get_dummies(pl_transfers['new_position'],drop_first=True) #we drop the first entry (Attacker in our case) to prevent multicollinearity 

#Adding the new position dummy variables into the dataset
pl_transfers = pd.concat([pl_transfers,pos_dummies], axis=1)

pl_transfers
```
**Output:**

![dummy variables](https://github.com/user-attachments/assets/7300e550-f820-4daa-ba1e-a13c8e6080cd)

### First attempt at building the regression model

After getting my dummy variables, I was ready to build my first ever unassisted linear regression model using the Ordinary Least Squares (OLS) method.
```
#importing the statsmodels library which will be used for our regression

import statsmodels.api as sm 

#importing the Standard scaler library so we can standardize our predictors (independent variabes).

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

#declaring the dependent(y) and independent(X) variables

X = pl_transfers[['year','age','transfer_period','Defender','Goalkeeper','Midfielder']]
y = pl_transfers['transfer_fee']

#standardizing the independent variables

X_scaled = scaler.fit_transform(X)

#Converting the scaled data back into a Dataframe with the original column names

X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

#Adding a constant to the model

x = sm.add_constant(X_scaled)

#fitting the OLS model and getting a summary of the model

model = sm.OLS(y,x).fit()
model.summary()
```
Running the code above, I was hit with the error message below
```
---------------------------------------------------------------------------
MissingDataError                          Traceback (most recent call last)
~\AppData\Local\Temp/ipykernel_21004/1701416372.py in <module>
      9 
     10 
---> 11 model = sm.OLS(y,X).fit()
     12 model.summary()

~\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
    870     def __init__(self, endog, exog=None, missing='none', hasconst=None,
    871                  **kwargs):
--> 872         super(OLS, self).__init__(endog, exog, missing=missing,
    873                                   hasconst=hasconst, **kwargs)
    874         if "weights" in self._init_keys:

~\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, weights, missing, hasconst, **kwargs)
    701         else:
    702             weights = weights.squeeze()
--> 703         super(WLS, self).__init__(endog, exog, missing=missing,
    704                                   weights=weights, hasconst=hasconst, **kwargs)
    705         nobs = self.exog.shape[0]

~\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, **kwargs)
    188     """
    189     def __init__(self, endog, exog, **kwargs):
--> 190         super(RegressionModel, self).__init__(endog, exog, **kwargs)
    191         self._data_attr.extend(['pinv_wexog', 'weights'])
    192 

~\anaconda3\lib\site-packages\statsmodels\base\model.py in __init__(self, endog, exog, **kwargs)
    235 
    236     def __init__(self, endog, exog=None, **kwargs):
--> 237         super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
    238         self.initialize()
    239 

~\anaconda3\lib\site-packages\statsmodels\base\model.py in __init__(self, endog, exog, **kwargs)
     75         missing = kwargs.pop('missing', 'none')
     76         hasconst = kwargs.pop('hasconst', None)
---> 77         self.data = self._handle_data(endog, exog, missing, hasconst,
     78                                       **kwargs)
     79         self.k_constant = self.data.k_constant

~\anaconda3\lib\site-packages\statsmodels\base\model.py in _handle_data(self, endog, exog, missing, hasconst, **kwargs)
     99 
    100     def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
--> 101         data = handle_data(endog, exog, missing, hasconst, **kwargs)
    102         # kwargs arrays could have changed, easier to just attach here
    103         for key in kwargs:

~\anaconda3\lib\site-packages\statsmodels\base\data.py in handle_data(endog, exog, missing, hasconst, **kwargs)
    670 
    671     klass = handle_data_class_factory(endog, exog)
--> 672     return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
    673                  **kwargs)

~\anaconda3\lib\site-packages\statsmodels\base\data.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
     85         self.const_idx = None
     86         self.k_constant = 0
---> 87         self._handle_constant(hasconst)
     88         self._check_integrity()
     89         self._cache = {}

~\anaconda3\lib\site-packages\statsmodels\base\data.py in _handle_constant(self, hasconst)
    131             exog_max = np.max(self.exog, axis=0)
    132             if not np.isfinite(exog_max).all():
--> 133                 raise MissingDataError('exog contains inf or nans')
    134             exog_min = np.min(self.exog, axis=0)
    135             const_idx = np.where(exog_max == exog_min)[0].squeeze()

MissingDataError: exog contains inf or nans
```
Basically, the error message was letting me know that there were missing values in my data. To solve this, I first had to identify where the missing values were coming from. I did this by running the following code
```
#identifying the columns with missing values
x.isna().sum()
```
I realized that there were two missing values in the age column. Next, I run the below code to see which rows had a error values and to see if anything could be done about them.
```
#identifying the exact rows with missing values in our age column
pl_transfers[pl_transfers['age'].isna()]
```
**Output:**

![missing player ages](https://github.com/user-attachments/assets/7e75a173-e775-4e60-a923-e728f3e17c01)

After a quick google search, I was able to correctly identify the ages both players were when their respective transfers went through. The code below:
```
#assigning the right ages to the players with missing age values
pl_transfers.loc[358,'age'] = 24
pl_transfers.loc[3649,'age'] = 28

#confirming that the changes made have been effected
pl_transfers[pl_transfers['age'].isna()]
```
**Output:**
![missing player age check](https://github.com/user-attachments/assets/6e1ed6dc-4504-4886-bc61-5be3b4e41f56)

## Second time's the charm 🙈

After fixing the issues with missing data, I run the model again and below is the summary of the model

```
#centering the year and age variables
#this is done so we can predict what the transfer fee is when the year and age variables are at the mean and not when they are 0.
pl_transfers['year_centered'] = pl_transfers['year']-pl_transfers['year'].mean()
pl_transfers['age_centered'] = pl_transfers['age']-pl_transfers['age'].mean()

#declaring the dependent(y) and independent(X) variables
C = pl_transfers[['year_centered','age_centered','transfer_period','Defender','Goalkeeper','Midfielder']]
d = pl_transfers['transfer_fee']

#Adding a constant to the model
c = sm.add_constant(C)

#fitting the OLS model and getting a summary of the model
model = sm.OLS(d,c).fit()
model.summary()
```
![OLS summary](https://github.com/user-attachments/assets/a720f828-ffc9-4b92-b54d-536a4589a75d)

### Interpretating the summary
Starting off with the R-squared value, this tells us how much of the variability of the dependent variable is explained by the model. A value of 0.184 tells us that 18.4% of the changes in the transfer fees is explained by the model. In this situation this isn't a good look for the model however, this makes sense considering that the model only looks at a players age, position, when the transfer was done and the year the transfer was done in. Important metrics like player performances in previous seasons and the league/team/country a player is being transferred from haven't been considered in the model.

Next thing to look at is the P-value. In the picture, that is denoted as P>|t|. This value let's us know if the coefficients are statistically significant in the model. For a coefficient to be statistically significant, it's P-value would have to be less than 0.05. In the summary, all of varaibles have a P-value less than 0.05 thus we can conclude that all of the independent varaibles in the model are statistically significant. 

Finally, it's time to interprete the coefficients.

- The constant coefficient of 8.299e+06 indicates that if we were given the mean year and player age and all other independent variables were 0, we would have the transfer fees being somewhere around 8,299,000 euros.
- 5.078e+05 is the coefficient for year and this tells us that as the years increase, the transfer fee of a player also increases by 507,800 euros.
- For age, we had a coefficient of -1.844e+05. This tells us that there is a negative relationship between a player's age and the transfer fee. As the age of a player increases, their transfer value decreases by 184,400 euros.
- The coefficient of the transfer period was -8.41e+05. As you will recall, this is a dummy variable with Summer as 0 (reference category) and Winter as 1. This means that the average winter transfer fee is about 841,000 euros less than the average summer transfer fee.
- Again, here we have another dummy variable with Attacker as our reference category. The coefficient of -2.675e+06 for Defender means that the transfer fee for a defender, on average, is 2,675,000 euros less than an Attacker's.
- For Goalkeepers, teams in the English Premier league would typically pay an average of 5,319,000 euros less than they would pay for an Attacker
- And for midfielders, on average, teams would pay 1,272,000 euros less than they would for an attacker




