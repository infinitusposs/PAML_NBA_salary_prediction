# PAML Final Project - NBA Salary Prediction

## Introduction
- **Motivation**: Build a machine-learning pipeline that can predict the salaries of NBA players based on their performance and background. 
While many players hire agents to negotiate their contracts, there are still cases where players receive salaries that do not match their performance. 
- **Technical focus**: We apply different regression models in this project and we attempt to select important features before fitting our models.
- **Impact**: Our hope with this project is to provide players and agents with a tool that can help them estimate their salaries more accurately.

## Background
In Papadaki and Tsagris’s work<sup>1</sup>, they pointed out that one of the current problem of NBA salary prediction is that many current solutions fail to check feature importance before training their models. In terms of the machine learning techniques, they transfer the problem into a classification problem and predicted the player salaries in percent of the teams payroll instead of salaries.

*[1] Papadaki, Ioanna, and Michail Tsagris. "Estimating NBA players salary share according to their performance on court: A machine learning approach." arXiv preprint arXiv:2007.14694 (2020).*

## Datasets
[NBA Players & Team Data](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data?select=NBA+Player+Stats%281950+-+2022%29.csv)
[NBA Players stats since 1950](https://www.kaggle.com/datasets/drgilermo/nba-players-stats?select=Seasons_Stats.csv)

## Team
Jiarong Chen (jc2924) \
Yufei Wang (yw569)

## Conclusion & Limitations
#### 1. The relationship between salaries and performance stats is not linear.
A non-linear model such as Random Forest or Deep Learning may give us better results.

| model_label       | dataset | mean_absolute_error | root_mean_squared_error | r2_score |
| -----------       | ----------- | -----------|----------- | -----------|
| Linear Regression | Training    | 2,394,051.1939| 3,201,046.5668| 0.1999|
| Linear Regression | Validation  | 2,346,751.219| 3,194,329.0371|0.1822 |
| Ridge             | Training    | 2,397,569.5221| 3,201,811.9231| 0.1995|
| Ridge             | Validation  | 2,352,241.4066|3,192,717.5129 |0.183 |
| Lasso             | Training    |2,410,388.9473 |3,201,477.4736 |0.1997 |
| Lasso             | Validation  | 2,366,040.6655| 3,192,275.9228|0.1997 |
| Poly              | Training    |2,333,015.7044 |3,122,270.4853 |0.2388 |
| Poly              | Validation  |2,293,413.425 |3,122,270.4853 | 0.2388|

#### 2. Average yearly salary for the next contract rather than for the next season.
Since NBA players usually sign contracts for multiple years and the salaries during contract period do not changeaccording to their performance (ignore bonuses and incentives, not included in the datasets), predicting the salary for the next contract based on its performance during the current contract will be more reasonable and more accurate.

##### Example:
One of the famour low-ball contract: Stephen Curry

| Season       | avg_PTS | avg_REB | avg_AST | Sign_newContract | Salary_Season | InflationAdjSalary
| -----------       | ----------- | -----------|----------- | -----------| -----------| -----------|
| 2009/10 | 17.49 |	4.45	|5.9	|<ul><li>[ ] </li></ul>	|2010/11|	3,961,199|
| 2010/11| 18.55 |	3.86	|5.84|	<ul><li>[ ] </li></ul>|	2011/12|	4,091,922|
| 2011/12 |14.73	|3.38|	5.31	| <ul><li>[x] </li></ul> |	2012/13	|5,111,683|
| 2012/13 |22.9	|4.03	|6.91	|<ul><li>[ ] </li></ul>	|2013/14	|12,547,181|
| 2013/14 |24.01	|4.28	|8.54	|<ul><li>[ ] </li></ul>|	2014/15	|13,214,370|
| 2014/15 |23.75	|4.26	|7.74	|<ul><li>[ ] </li></ul>|	2015/16	|14,118,828|
| 2015/16 |30.06	|5.44	|6.67	|<ul><li>[ ] </li></ul>|	2016/17	|14,891,108|
| 2016/17 |25.3	|4.47	|6.62	| <ul><li>[x] </li></ul>|	2017/18	|41,953,914|

## Run the application:
```
streamlit run final_project.py
```
