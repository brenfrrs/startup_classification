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

The original 39 columns were as follows:

- permalink 
- name
- homepage url
- category list
- market
- funding total usd
- status
- country code
- state code
- region
- city
- funding rounds
- founded at
- founded month
- founded quarter
- founded year
- frist funding at
- last funding at
- seed
- venture
- equity crowdfunding
- undisclosed
- convertible note
- debt financing
- angel
- grant
- private equity
- post ipo equity
- post ipo debt
- secondary market
- product crowdfunding
- round A
- round B
- round C
- round D
- round E
- round F
- round G
- round H

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
