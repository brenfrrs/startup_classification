# Predicting Startup Acquisition 

![startup.jpeg](./images/startup_acquisitions_blue.jpeg
)

**Authors**: [Brendan Ferris](mailto:brendanfrrs@gmail.com), [Michael Wirtz](mailto:michaelwirtz88@gmail.com)

## Overview

This project analyzes the needs of Butterfly Ventures, a micro VC fund that is looking to predict on the acquisition of startups. In an effort to model this problem, we collected a dataset of startups that fell into any one of the following three categories: closed, operating or acquired. In an effort to minimize the false negatives, we chose precision to be our target metric. Our baseline model using Logistic Regression had a precision score of 0.1. Our final and best model was a Random Forest model that had a precision score of 0.33. 

## Business Problem

Butterfly Ventures is small VC fund that is low on capital. Because of their limited funds, they are looking for a way to better filter companies in the hopes of making the most of their investments. They are aware of the following statistics: 75% of venture-backed startups fail. Under 50% of businesses make it to their fifth year. 33% of startups make it to the 10-year mark. Only 40% of startups actually turn a profit. Given this knowledge, Butterfly Ventures is targeting startups that they believe have the best opportunity at acquisition, a sure-fire way for investment profits. For this purpose, they have hired a group of data scientists to create a model predicting whether or not a startup will be acquired. 


## Data 

In order to help Butterfly Ventures, we used a [Kaggle dataset](https://www.kaggle.com/arindam235/startup-investments-crunchbase) to use in our modeling process. The three given classification in the dataset were "closed," "operating," and "acquired."

<table>
<tr>
<th> Features </th>
</tr>
<tr>
<td>

<ul>
<li> <b>funding_total_usd</b>: how much money did the company raise in total.</li>
<li> <b>seed</b>: early stage investments meant to support the business until it can generate cash of its own.</li>
<li> <b>venture</b>: money raised through venture capital.</li>
<li> <b>equity_crowdfunding</b>: equity sold via crowdfunding.</li>
<li> <b>undisclosed</b>: mondey raised through undisclosed means.</li>
<li> <b>convertible_note</b>: short-term debt that converts into equity</li>
<li> <b>debt_financing</b>: company raises money by selling debt instruments to investors. Unlike equity financing, this form of financing must be paid back.</li>
<li> <b>angel</b>: Amount of capital raised by an individual investor in exchange for convertible debt or ownership equity. </li>
<li> <b>grant</b>: The amount of money raised through grants.</li>
<li><b>private_equity</b>: funds raised by private-equity firms, venture capital firms, or angel investors.</li>
<li><b>round_A</b>: the value of the company is usually determined during the initial round of funding.</li>
<li><b>round_B</b>: funds raised during the second round of funding, after a company has reached certain milestones.</li>
<li><b>days_from_founding_to_funding</b>: the amount of days that passed between the companies founding and when they were first able to secure funding.</li>
<li><b>time_between_first_and_last_funding</b>: the amount of days that passed between the first time the company recieved funding and the last time the company recieved funding.</li>
<li><b>month_<i>X</i></b>: the month that the company was founded. </li>
<li><b>founded_quarter_<i>X</i></b>: the quarter (Q1,Q2,Q3,Q4) that the company was founded.</li>
<li><b>state_code_<i>X</i></b>: the state the company was founded in.</li>
<li><b>founded_year_<i>X</i></b>: the year the company was founded.</li>
<li><b>url_ending_<i>X</i></b>: the domain name ending of the company website (.com, .org, etc)</li>
<li>funding_rounds_<i>X</i></li>
<li><b>country_code_USA</b>: if the company was founded in the USA.</li>

</ul>

</td>
</tr>
</table>

<table>
<tr>
<th> Target </th>
</tr>
<tr>
<td>

<ul>
<li> <b>acquired (1)</b>: was the company acquired.</li>
<li> <b>not acquired (0)</b>: companies that have either closed or are still operating.                 </li>
</ul>

</td>
</tr>
</table>

Because certain values possess overly predictive power, they were dropped from the models. Those columns are as follows: 

- post ipo equity
- post ipo debt
- round C
- round D
- round E
- round F
- round G
- round H

## Methods

Overall, this project analyzes the given dataset information to maximize the precision metric of our models. 

In order to get the most out of our column data, we dummied all of the categorical columns. We presumed that the category list column would be the most beneficial to our model, given that it would be able to classify each startup specifically into business-type categories. 

Because there was high class imbalance, we upsampling and downsampling techniques to even out the True and False classes.

For our logistic regression models we also standardized the continous values because they were throwing off the models with their extremely different values. 

## Results

Our results had two stages: 

First, we trained our models and predicted on the test set. Because our test set was balanced, we received some pretty impressive results. Our models showed precision scores close to 0.9. 

Second, we predicted on the holdout set. The results here were vastly different. We beleive that this is because the class imbalance was just so high. Due to this fact, our precision score fell dramatically below 0.5, with our best model having a precision score on the holdout set of 0.32. 

## Conclusion

The conclusions that can be drawn given our results are the following:

- Predicting a quality precision score on startup acquisition requires way more data
- There are only small tangible differences that make the difference between an acquired startup and a startup that is not acquired
- Given the complexity of this task, there is a vast amount of data required. If a high precision score was possible given the small dataset that we used, all VC firms would be wildly successful.


### Next Steps

- Use a more interpretable model to find out which characteristics are most highly correlated with startup acquisition
- scrape data on startup management to get an indication of how that can affect acquisition 
- Overall: get more data! 


## For More Information

See the full analysis in the [Jupyter Notebook](./code_success_movie.ipynb) or review this [presentation](./slides_successful_movie.pdf).

For additional info, contact Brendan Ferris or Michael Wirtz at
[brendanfrrs@gmail.com](mailto:brendanfrrs@gmail.com) and [michaelwirtz88@gmail.com](mailto:michaelwirtz88@gmail.com), respectively

## Repository Structure
