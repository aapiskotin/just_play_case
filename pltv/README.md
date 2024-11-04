# Case Study 1: Predicting Lifetime Value (LTV) for JustPlay Users

## 0. Preliminary Analysis Results ([0_EDA.ipynb](src/0_EDA.ipynb))

The preliminary analysis of the provided dataset showed a highly synthetic nature
of the data:
1. The targets distribution is too unnatural. The majority of the samples falls into the minimum and maximum value of the target 
   with some fluctuations in between, while the natural distribution of LTV
   *in my experience* should follow a log-normal (in some cases normal) distribution.
2. The targets distribution is extremely stable in time, which is unnatural given
   the time frame of the year provided in the dataset.
3. The number of acquistions per day is also to stable for real world scenario.
4. The date of scoring (<-> the date of the feature collection, usually called
   `sample_date`) is not provided. 

Given the observations above, I am going to fluctuate from the framework that
would be applicable for the real-world case and will simply try to maximize the
accuracy of the predictions given the features and targets provided.

## 1. The Solution

Given the limited time to solve the problem, I am going to describe the steps
I want to take, while not all the steps are going to be implemented in the current
version of the solution.

### 1.1. The validation framework

Given the stability of the targets in time, the validation framework doesn't
include the validation of the re-train routines, which is a common practice in
the real-world scenario.

#### 1.1.1. Metrics

The main metric to evaluate the solution is going to be 
[`WAPE` (Weighted Absolute Percentage Error)](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html#metrics-WAPE) - 
this metric gives us interpretable error in terms of the percentage of the
error in the target value while being robust to the distribution of the target.

The secondary metrics to use are going to be `MAE` (Mean Absolute Error) and
`RMSE` (Root Mean Squared Error).

#### 1.1.2. Validation Dataset

The evaluation will be performed over the holdout OOT (out-of-time) dataset
of the "last" (by acquisition date) 30% of the samples.

#### 1.1.3. Experiment Tracking 

The experiment tracking is going to be performed using Weight & Biases platform.

### 1.2. The Model

The experimentation plan:

- [X] Baseline: prediction by means(or medians) over the train dataset. This is the simplest
  model to beat. However, we can build our validation framework on top of it.
- [X] Classification Model: given the distribution of the targets, the binary 
      classification over min and max values could bring us good accuracy.
- [X] Prediction of the intermediate samples via the weightening min and max 
      values by the probabilities predicted by the classification model.
- [ ] Regression Model: The intermediate values can be predicted by the regression
      over the residuals of the classification model.

The intermediate steps of the model evaluation (like the ones that will be used
in the Feature Selection routine) will be performed on the Cross-Validation
over the train dataset. This will make our validation framework more robust
to the overfitting (which can occur given that every metric has some non-zero
type I error rate).

The Feature Selection step is crucial in the given task, since the number of features
is high and some of them seems to be prone to the `leaking` of the target information.

## 2. Results

1. The Validation Framework is implemented and tested over various models. 
    The Framework gives us an opportunity to iterate over various hypothesis
    and models in consistent way.
   - The `Validator` class and the `Model` interface are presented in the
     [validate.py](src/validate.py) file.
   - The experiments tracked via Weight & Biases are presented 
   [here](https://wandb.ai/justplay-case/justplay-pltv/table?nw=nwuseraapiskotinge)
     (copy in Google Sheets [here](https://docs.google.com/spreadsheets/d/1stcLaI7b4AUn29uSwcLOW7-keo_Z7yC01tK_GUFzu8M/edit?usp=sharing))
2. The best model iteration is Classification Model with interploation via probabilities.
   - The experimentation results are presented in the [2_clf_model.ipynb](src/2_clf_model.ipynb) notebook.
   - The model shows good results on the given dataset, though some steps can be done to
     improve the accuracy and robustness of the model.
   - The interpretation of the model follows common sense (Details are presented in the notebook).
3. The Solution presented is far from production ready, but it gives us a good starting point
   to build the real-world solution.

### Future Steps

Given the limitations of the time and data, the following steps can be done to 
improve the solution:

1. While the classification setup seems to work pretty well on the
   target with the given distribution, some more sophisticated regression models
    to be stacked upon the classification model can be tested in order to improve
    the accuracy of the model.
2. The feature selection routine wasn't implemented in the current version of the solution.
     However, it can become crucial in the real-world scenario to avoid the overfitting
     I have encountered in the experiments.
3. Nothing done in the production-readiness and scalability direction, while there are
    lots of steps to be implemented to make the solution ready for the production
   usage.
