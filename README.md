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
After importing the relevant libraries and the cleaned dataset into python, it was time to start building the regression model.

### Creating dummy variables
At this point, I already knew what my dependent and independent variables were going to be. The dependent variable, which is the model I was going to try to predict was the transfer_fee and the independent variables (predictors) were the year,age,new_position and transfer_period.

However, there was a small problem with the dependent variables. There were two columns there that had categorical data. the new_position and the transfer_period varaibles
