# Python implementation of independence tests.

Hoeffding_independence.py is written based on two papers. 

The calculation of Hoeffding's independence test statistic is written according to: Hoeffding, Wassily. "A non-parametric test of independence." The annals of mathematical statistics (1948): 546-557.

The asymptotic p values are calculated according to: Wilding, Gregory E., and Govind S. Mudholkar. "Empirical approximations for Hoeffding’s test of bivariate independence using two Weibull extensions." Statistical Methodology 5.2 (2008): 160-170.

The function Hoeffding_independece_test provides several options. A simple test could be run as follows:

```python
test = Hoeffding_independece_test('test')
Dn, p_value = test(X, y)
``` 

Please keep in mind, this is only an approximation of the null distribution of nDn.
