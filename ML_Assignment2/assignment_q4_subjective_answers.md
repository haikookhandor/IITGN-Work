# Plots for Classification Data

## n_jobs = 1, without Parallelism:

N_estimators:(3)
<p align="left">
  <img width="400" src="figures\BaggingClassifier_jobs1.png">
</p>

Common Decision Surface:

<p align="left">
  <img width="400" src="figures\Common_Decision_Surface_jobs1.png">
</p>


## n_jobs = 3, with Parallelism:

N_estimators:(4)
<p align="left">
  <img width="400" src="figures\BaggingClassifier_manyjobs.png">
</p>

Common Decision Surface:

<p align="left">
  <img width="400" src="figures\Common_Decision_Surface_manyjobs.png">
</p>

Timing Analysis:

<p align="left">
  <img width="400" src="figures\q4_timing_analysis.png">
</p>

Observations:

For N being low, we can see that the time taken by one job is less than that of many jobs. This is bevause since the number of datapoints are less, the timing comparison cannot be done effectively as the effects of parallelism are apparent as the number of data points increases.

For N being high, we can see that now the time taken by parallelism i.e. many jobs is now less than that of the one job which is what the expected behavior is.