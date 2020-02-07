# Python implementation of independence tests.

## Partial correlation

A very naive implementation of partial correlation between Xi and y, while controlling for other columns in X.

## Hoeffding's independence test

The calculation of Hoeffding's independence test statistic is written according to: Hoeffding, Wassily. "A non-parametric test of independence." The annals of mathematical statistics (1948): 546-557. <https://www.jstor.org/stable/pdf/2236021.pdf>

The asymptotic p values are calculated according to: Wilding, Gregory E., and Govind S. Mudholkar. "Empirical approximations for Hoeffding’s test of bivariate independence using two Weibull extensions." Statistical Methodology 5.2 (2008): 160-170. <https://doi.org/10.1016/j.stamet.2007.07.002>

The function Hoeffding_independece_test provides several options. A simple test could be run as follows:

```python
test = Hoeffding_independece_test('test')
Dn, p_value = test(X, y)
``` 

Please keep in mind, this is only an approximation of the null distribution of nDn.

## Distance correlation

The sample distance correlation function is written based on the simplified formula in Székely, Gábor J., and Maria L. Rizzo. "Partial distance correlation with methods for dissimilarities." The Annals of Statistics 42.6 (2014): 2382-2412. <https://projecteuclid.org/euclid.aos/1413810731>
