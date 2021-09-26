**2019-38914 Hyuntae Choi**

# 

**Suppose that you have data on J cars in a single market. Show that from the demand-side alone you can not simultaneously estimate coefficients on the group- specific constants together with the within-group correlation coefficient, $\sigma$. What function of these parameters is identified?**

(1) $s_j = \bar{s}_{j/g} \cdot \bar{s}_g$ implies that  $\log(s_j) = \log(\bar{s}_{j/g}) + \log(\bar{s}_g)$.

(2) $\log(s_j) - \log(s_0) = x_j\beta-\alpha p_j+\sigma\log(\bar{s}_{j/g})+\xi_j.$

Plugging (1) into (2), $$\log(s_j) - \log(s_0) = x_j\beta-\alpha p_j+\sigma(\log(s_j)-\log(\bar{s}_g))+\xi_j=x_j\beta-\alpha p_j+\sigma(\log(s_j)-\Sigma_gd_{jg}\log(\bar{s}_g))+\xi_j,$$

where $d_{jg}$ is a group dummy variable for group $j$. $d_{jg}=1$ if product $j$ is in group $g$, $d_{jg}=0$, otherwise .

Then $$(3) \log(\bar{s}_{j/g})=[\log(s_j)-\log(s_0)]-\Sigma_gd_{jg}\cdot(\log(\bar{s}_g))+\log(s_0),$$ which implies the existence of linear dependence between $\log(\bar{s}_{j/g}), \log(s_j)-\log(s_0), d_{jg}$.

Theoretically, in the process of deducing (2), the random coefficient for group dummy variables are integrated so it is absurd to use group dummy variables in the model again.

The equation (2),

(2) $\log(s_j) - \log(s_0) = x_j\beta-\alpha p_j+\sigma\log(\bar{s}_{j/g})+\xi_j$

is used for estimating demand parameters $\alpha$ and $\sigma$. According to the collinearity problem, group dummy variables are excluded from $x_j$. $x_j$ includes a measure of reliability, and a dummy variable indicating whether the car is made by a domestic producer.

# 

**Using the data that I make available to you, estimate the identified demand parameters using (a) OLS and (b) 2SLS with the *excluded* cost-side variables. Discuss your results**.

## Preparation

$\log(s_j), \log(s_0)$, and  $\log(\bar{s}_{j/g})$ are calculated to use equation (2) for estimation.


```python
# import library
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
```


```python
# load dataset
dfAuto = pd.read_csv("auto1.txt", sep='\t', header=None, names=['Price', 'Qty', 'Class', 'Country', 'Reliability'])
```


```python
# initial check
print(dfAuto.head().to_markdown())
```

    |    |   Price |    Qty |   Class |   Country |   Reliability |
    |---:|--------:|-------:|--------:|----------:|--------------:|
    |  0 |   9.143 | 83.599 |       3 |         2 |             5 |
    |  1 |  18.944 | 53.666 |       5 |         2 |             5 |
    |  2 |  16.029 |  1.549 |       3 |         3 |             2 |
    |  3 |  14.461 |  4.402 |       2 |         3 |             4 |
    |  4 |  18.355 |  3.38  |       2 |         3 |             4 |


Calculate $s_j$, $\bar{s}_{j/g}$, $s_0$, and $\log(s_j)-\log(s_0)$ using $M$. Since Qty is measured in thousands, $M$ is adjusted to match the unit. In addition make dummy variable *domestic* that represents domestic producer.

(4) $s_j=\text{Qty}_j/M$

(5) $\bar{s}_{jg}=\text{Qty}_j/\Sigma_{j\in g}\text{Qty}_j$

(5) $s_0 = 1 - \Sigma_j s_j$

I used `python` for implementation here and attached `R` code in the Appedix.


```python
M = 95500
dfAuto['sj'] = dfAuto['Qty'] / M
dfAuto['sjg'] = dfAuto.groupby(['Class'])['Qty'].apply(lambda x: x/x.sum())
s0 = 1 - dfAuto['sj'].sum()
dfAuto['lnsjMinuslns0'] = np.log(dfAuto['sj']) - np.log(s0)
dfAuto['lnsjg'] = np.log(dfAuto['sjg'])
dfAuto['domestic'] = dfAuto.apply(lambda x: 1 if x['Country'] == 1 else 0, axis=1)
```


```python
print((dfAuto[['Price', 'Qty', 'Reliability', 'sj', 'sjg', 'lnsjMinuslns0', 'lnsjg', 'domestic']].head().to_markdown()))
```

    |    |   Price |    Qty |   Reliability |          sj |         sjg |   lnsjMinuslns0 |    lnsjg |   domestic |
    |---:|--------:|-------:|--------------:|------------:|------------:|----------------:|---------:|-----------:|
    |  0 |   9.143 | 83.599 |             5 | 0.000875382 | 0.0773888   |        -6.94538 | -2.55891 |          0 |
    |  1 |  18.944 | 53.666 |             5 | 0.000561948 | 0.0756022   |        -7.38863 | -2.58227 |          0 |
    |  2 |  16.029 |  1.549 |             2 | 1.62199e-05 | 0.00143393  |       -10.9338  | -6.54734 |          0 |
    |  3 |  14.461 |  4.402 |             4 | 4.60942e-05 | 0.00127589  |        -9.88936 | -6.66411 |          0 |
    |  4 |  18.355 |  3.38  |             4 | 3.53927e-05 | 0.000979674 |       -10.1535  | -6.92829 |          0 |


## OLS Results

The table below shows the OLS result. The model is
$$(6) \log(s_j) - \log(s_0) = \beta_0 + \beta_1\text{reliablilty}_j + \beta_2\text{domestic}_j + \alpha\text{Price}+ \sigma\log(\bar{s}_{j/g}) + \xi_j.$$


```python
mdl_1 = smf.ols(formula = "lnsjMinuslns0 ~ Reliability + domestic + Price + lnsjg", data = dfAuto)
print(np.round(mdl_1.fit().summary2().tables[1], 5).to_markdown())
```

    |             |    Coef. |   Std.Err. |         t |   P>|t| |   [0.025 |   0.975] |
    |:------------|---------:|-----------:|----------:|--------:|---------:|---------:|
    | Intercept   | -3.87388 |    0.23058 | -16.8005  | 0       | -4.3302  | -3.41757 |
    | Reliability | -0.018   |    0.03341 |  -0.53882 | 0.59096 | -0.08411 |  0.04811 |
    | domestic    |  0.0416  |    0.09629 |   0.43199 | 0.66649 | -0.14896 |  0.23216 |
    | Price       | -0.03375 |    0.00417 |  -8.09847 | 0       | -0.042   | -0.02551 |
    | lnsjg       |  0.89532 |    0.03524 |  25.4093  | 0       |  0.82559 |  0.96505 |


Coefficient for Reliability is negative($-0.018$) which is hard to accept since it is reasonable to guess that an increase in the reliability score in product $j$ would increase the market share of that product. Also Berry pointed out the endogeneity problem associated with Price($p$) and $\log(\bar{s}_{\cdot/g})$. This naturally leads to 2SLS using IVs for $p$ and $\log(\bar{s}_{\cdot/g})$

## 2SLS Results

Candidates for IVs are *excluded* cost side variables("Class", "Country"). In addition, BLP IVs could be candidates. For product $j$, the characteristics $x_i$ of product $i$ are uncorrelated with unobserved characteristics $\xi_j$ of $j$ but highly correlated with $p_j$ in the oligopoly setting. BLP IV is calculated as the sum of reliability except for that of product j itself, denoted "z" and the sum within the group except for that of product $i$ denoted "zg".

(7) $z_j = \Sigma_{i, i\neq j} \text{Reliability}_i$

(8) $zg_j = \Sigma_{i\in g\\i\neq j} \text{Reliability}_i$


```python
# make "z" and "zg"
dfIV = dfAuto.copy()
for i in range(len(dfAuto)):
    dfIV.loc[i, 'z'] = dfAuto.drop(i)['Reliability'].sum()
s = dfAuto.groupby(['Class']).sum()[['Reliability']].add_prefix('sum_within_group_')
dfIV = dfIV.join(s, 'Class')
dfIV.columns = ['Price', 'Qty', 'Class', 'Country', 'Reliability', 'sj', 
                'sjg', 'lnsjMinuslns0', 'lnsjg', 'domestic', 'z', 'zg']
```


```python
print(dfIV[['Price', 'Qty', 'Reliability', 'z', 'zg']].head().to_markdown())
```

    |    |   Price |    Qty |   Reliability |   z |   zg |
    |---:|--------:|-------:|--------------:|----:|-----:|
    |  0 |   9.143 | 83.599 |             5 | 403 |   93 |
    |  1 |  18.944 | 53.666 |             5 | 403 |   66 |
    |  2 |  16.029 |  1.549 |             2 | 406 |   93 |
    |  3 |  14.461 |  4.402 |             4 | 404 |  101 |
    |  4 |  18.355 |  3.38  |             4 | 404 |  101 |


*Excluded* cost side variables are categorical. Dummy variables are needed since it is not suitable to use them directly. They are denoted as ("class_1", "class_2", "class_3", "class_4", "class_5") and ("country_1", "country_2", "country_3").


```python
dfIV = pd.concat([dfIV, pd.get_dummies(dfIV['Class']).add_prefix('class_')], axis=1)
dfIV = pd.concat([dfIV, pd.get_dummies(dfIV['Country']).add_prefix('country_')], axis=1)
```


```python
print(dfIV[['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'country_1', 'country_2', 'country_3']].head().to_markdown())
```

    |    |   class_1 |   class_2 |   class_3 |   class_4 |   class_5 |   country_1 |   country_2 |   country_3 |
    |---:|----------:|----------:|----------:|----------:|----------:|------------:|------------:|------------:|
    |  0 |         0 |         0 |         1 |         0 |         0 |           0 |           1 |           0 |
    |  1 |         0 |         0 |         0 |         0 |         1 |           0 |           1 |           0 |
    |  2 |         0 |         0 |         1 |         0 |         0 |           0 |           0 |           1 |
    |  3 |         0 |         1 |         0 |         0 |         0 |           0 |           0 |           1 |
    |  4 |         0 |         1 |         0 |         0 |         0 |           0 |           0 |           1 |


### Stage 1

$p$ and $\log(\bar{s}_{\cdot/g})$ is regressed on instrument variables and their fitted value is calculated as $\hat{p}$("p_hat") and $\widehat{\log(\bar{s}_{\cdot/g})}$("lnsjg_hat"). The precise models are
$$ (9) p_j = \beta_0 + \beta_1 * \text{class}_{1j} + \beta_2 * \text{class}_{2j} + \beta_3 * \text{class}_{3j} + \beta_4 * \text{class}_{4j} + \beta_5 * \text{z}_j+ \epsilon_j$$
$$ (10) \log(\bar{s}_{j/g}) = \beta_0 + \beta_1 * \text{country}_{1j} + \beta_2 * \text{country}_{2j} + \beta_3 * \text{zg}_{j} + \eta_j.$$

I implicitly assume that Price is highly correlated with Class characteristic as a supply side variable and also with BLP IV `z`(equation (9)). In equation (10) I assumed that group share variable $log(\bar{s}_{j/g})$ is correlated with Country, and BLP IV `zg`.


```python
p = dfIV[['Price']]
X = dfIV[['class_1', 'class_2', 'class_3', 'class_4', 'z']]
X = sm.add_constant(X)
dfIV['p_hat'] = np.dot(np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)), p)
```

    /Users/choidamian/opt/anaconda3/envs/econometrics/lib/python3.8/site-packages/statsmodels/tsa/tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)



```python
lnsjg = dfIV[['lnsjg']]
X = dfIV[['country_2', 'country_3', 'zg']]
X = sm.add_constant(X)
dfIV['lnsjg_hat'] = np.dot(np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)), lnsjg)
```


```python
print(dfIV[['Price', 'lnsjg', 'p_hat', 'lnsjg_hat']].head().to_markdown())
```

    |    |   Price |    lnsjg |    p_hat |   lnsjg_hat |
    |---:|--------:|---------:|---------:|------------:|
    |  0 |   9.143 | -2.55891 | 16.3713  |    -3.97747 |
    |  1 |  18.944 | -2.58227 | 27.7837  |    -3.75273 |
    |  2 |  16.029 | -6.54734 | 16.7965  |    -5.17992 |
    |  3 |  14.461 | -6.66411 |  9.70797 |    -5.24651 |
    |  4 |  18.355 | -6.92829 |  9.70797 |    -5.24651 |


### Stage 2

The second stage model is
$$ (11) \log(s_j) - log(s_0) = \beta_0 + \beta_1\text{reliablilty}_j + \beta_2\text{domestic}_j + \alpha\hat{p}+ \sigma\widehat{\log(\bar{s}_{j/g})} + \xi_j.$$


```python
mdl = smf.ols(formula = "lnsjMinuslns0 ~ Reliability + domestic + p_hat + lnsjg_hat", data = dfIV)
print(np.round(mdl.fit().summary2().tables[1], 5).to_markdown())
```

    |             |    Coef. |   Std.Err. |        t |   P>|t| |   [0.025 |   0.975] |
    |:------------|---------:|-----------:|---------:|--------:|---------:|---------:|
    | Intercept   | -4.92479 |    1.30994 | -3.75954 | 0.00026 | -7.51713 | -2.33245 |
    | Reliability |  0.12493 |    0.0907  |  1.37744 | 0.17082 | -0.05456 |  0.30443 |
    | domestic    |  0.44919 |    0.37996 |  1.18221 | 0.23935 | -0.30274 |  1.20113 |
    | p_hat       | -0.04245 |    0.01412 | -3.00588 | 0.0032  | -0.0704  | -0.0145  |
    | lnsjg_hat   |  0.76124 |    0.25341 |  3.004   | 0.00322 |  0.25975 |  1.26273 |


The above shows the result of the stage 2 regression. $est. \alpha = -0.042$, and $est. \sigma = 0.761$. The coefficient for Reliability is about $0.125$ which is fairly large compared to $\alpha$. Lastly the coefficient of  domestic is estimated to $0.45$ which shows the large impact of the country of producing firm.

# 

## Producer sice first-order conditions

In a nested logit model, 

(12) $s_j(\delta, \sigma)=\bar{s}_{j/g}(\delta, \sigma)\bar{s}_g(\delta, \sigma)=\frac{e^{\delta_j/(1-\sigma)}}{D^\sigma_g[\Sigma_gD^{(1-\sigma)}_g]}$.

From (12),

(13) $\partial s_j /\partial \delta_j= \frac{1}{(1-\sigma)}s_j[1-\sigma\bar{s}_{j/g}-(1-\sigma)s_j]$.

FOC implies that

(14) $p_j=\bar{c}(x_j)+\frac{1}{\alpha}[s_j(\partial s_j/\partial \delta_j)] + \omega_j$.

Plugging (13) into (14)

(15) $p_j = w_j\gamma + [\frac{(1-\sigma)}{\alpha} / [1-\sigma\bar{s}_{j/g}-(1-\sigma)s_j]] + \omega_j$

The above relationship in the data between prices, product shares, and group shares will help to identify the substitution parameter, $\sigma$.

## (2) data on a cross-section or time-series of markets

In a cross section or time series markets with differing populations, population is a potential instrument for output quantities, $q_j$. When there is information on a number of markets, M can be parameterized as depending on market-level data (such as population) that vary across markets and that affect the aggregate level of output

# Appendix


```python

```
