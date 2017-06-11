# Predict which voter-eligible Colorado citizens actually voted in the 2016 US presidential election.

This is the Spring 2017, Harvard Statistics 149: Generalized Linear Models prediction contest/course project.

The goal of this project is to use the modeling methods you learned in Statistics 149 (and possibly other related methods) to analyze a data set on whether a Colorado voting-eligible citizen ended up actually voting in the 2016 US election. These data were kindly provided by moveon.org. The competition can be found [here](inclass.kaggle.com/c/who-voted) and ended April 30, 2017, at 10pm EDT.

To predict which Colorado voters voted in the 2016 election, we first performed [exploratory data analysis](who-voted_EDA.ipynb) followed by exploration of [features](who-voted_features.ipynb) that could potentially be useful for the prediction task. Exploration of the data revealed a large number of observations were missing data, therefore our next step was to [impute](who-voted_impute.ipynb) these missing data. We next fit [models](who-voted_modeling.ipynb) to predict voter turnout, and in our [final analysis](who-voted_final.ipynb) investigated important features that determined whether an individual was likely to vote or not.
