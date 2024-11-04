# Case Study 2: Accelerated A/B Testing and Automation for JustPlay

## The Solution

The acceleration of A/B testing is a huge project with many directions involved.
This directions can be summarized as follows:
1. **Split System**: The correct splitting system that allows multiple parallel tests
    without their interference is a huge helper in accelerating the tests. 
2. **Dynamic User Allocation**: This part can be included in the Split System,
    however, the *dynamic* part of this system makes it a separate entity, it is
    often task specific rather than a general solution.
3. **Statistical Criteria**: The huge impact on the accelartion of the tests can be
    made by the usage of the correct statistical criteria with some variance reduction
    techniques. A huge advantage of this part is that it is very generalizable.

### 1. Split System

Even though there is an obvious advantage in development a sophisticated split system,
the current solution omits this part, since designing of such a system would require
a lot of time and access to huge amounts of real historical data.

### 2. Dynamic User Allocation

It seems to me that the dynamic user allocation is very task specific part of the A/B
testing system. Which is why in the situation of limited time, there is only general
solution presented in [mab.py](src/mab.py) file. 

The solution is a plain usage of the
Multi-Armed Bandit algorithm with the Thompson Sampling strategy (which is known to
be SOTA in such tasks).

### 3. Statistical Criteria

To solve the task of developing a statictical criteria, the first step is the implementation
of the testing framework.

### Testing Framework

For the algorithm to be usable as a statistical test, the following criteria must be met:
1. **P-Value test**: The p-value of the criterion must follow a uniform distribution
   over the tests where no effect is present.

Given the dataset, the testing framework consists of the following steps:
1. Drawing bootstrap samples from one of the groups, we collect the distributions 
   of the p-values. The distribution of the p-values is tested for uniformity via
    the Kolmogorov-Smirnov test.
3. Given the real test data, calculate the p-values in relation to the number of samples
   drawn in each group. The smaller sample size must be drawn for the test to be significant
   the better.

### The Results [1_test.ipynb](src/1_test.ipynb)
0. The [Testing Framework](src/validate.py) is implemented which allows further testing
    of various statistical criteria to be applied to the real-world data.
1. The following statistical tests were tested:
    - Student's T-Test
    - Kolmogorov-Smirnov Test
    - Mann-Whitney U-Test
    - Bootstrap Test
    - Stratifiet T-Test
2. The first 3 tests mostly were used as the baseline to evaluate the testing framework.
    The results that were obtained from the tests met expectations: while the T-Test and
    U-test are suitable for the data presented, the KS Test failed on such data
    (which is expected given the binary metric).
3. The bootstrap test showed similar power as the T-Test and U-Test, though it is
    much more time consuming to calculate.
4. The Stratified T-Test also haven't shown any significant improvement over the
    T-Test and U-Test. It seems that the reason of such behavior is the 
    synthetic nature of the dataset - the conversion rates through the strata
    are mostly the same, which makes the stratification useless.
5. The Stratified T-Test criterion is [packed](src/stat_test.py) into the interface 
   with two basic methods - `get_sample_size` and `get_pvalue`

### Future Steps
1. Stratification is only of many variance reduction techniques that can be used
    to improve the power of the test. Some of them like CUPED, linearization and many
    could have also been implemented and tested via the provided framework.
2. The solutions presented in .py files aren't really usable in the real-world scenario.
    The next step would be to pack the solutions into the API and test them on the real-world
    data.
