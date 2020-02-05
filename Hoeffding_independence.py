import numpy as np

C_func = lambda x: np.int64(x>=0)
def primer(X):
    n = len(X)
    X_matrix = X*np.ones([n,n])
    primer_x = C_func(X_matrix.T - X_matrix)
    return primer_x
def Hoeffding_Dn(X, Y):
    n = len(X)
    primer_x = primer(X)
    a_alpha = np.sum(primer_x, 1) - 1
    primer_y = primer(Y)
    b_alpha = np.sum(primer_y, 1) - 1
    c_alpha = np.sum(primer_x * primer_y, 1) - 1
    A = np.sum(a_alpha*(a_alpha-1)*b_alpha*(b_alpha-1))
    B = np.sum((a_alpha-1)*(b_alpha-1)*c_alpha)
    C = np.sum(c_alpha*(c_alpha-1))
    return (A - 2*(n-2)*B + (n-2)*(n-3)*C)/(n*(n-1)*(n-2)*(n-3)*(n-4))
def power(x, alpha):
    return np.exp(alpha*np.log(x))
def generalized_Weibull_params(n):
    sigma_g = 0.1435
    lambda_g = -7.26
    gamma_g = 0.01266
    mu_g = -0.1537
    sigma_gn = sigma_g + 0.0385 * np.exp(-0.0019*n) + 0.38*np.exp(-0.174*n)
    lambda_gn = lambda_g/(1 + 2.09*np.exp(-0.1*n) - 0.3*np.exp(-0.0025*n))
    gamma_gn = gamma_g + 0.0037*np.exp(-0.0114*n) + 0.023*np.exp(-0.088*n)
    mu_gn = mu_g*(1 + 0.0253*np.exp(-0.0002*n) + 2.43*np.exp(-0.168*n))
    return sigma_gn, lambda_gn, gamma_gn, mu_gn
def pdf_generalized_Weibull(sigma_gn, lambda_gn, gamma_gn, mu_gn):
    def fn(x):
        x_ = (x - mu_gn)/sigma_gn
        x__ = power(x_, 1/gamma_gn)
        return (1/(gamma_gn*sigma_gn))*power(1-lambda_gn*x__, 1/lambda_gn-1)*x__/x_
    return fn  
def cdf_generalized_Weibull(sigma_gn, lambda_gn, gamma_gn, mu_gn):
    Fn = lambda x: 1 - power((1 - lambda_gn*power((x - mu_gn)/sigma_gn, 1/gamma_gn)) , 1/lambda_gn)
    return Fn
def quantile_generalized_Weibull(sigma_gn, lambda_gn, gamma_gn, mu_gn):
    Qn = lambda x: mu_gn + sigma_gn * power(((1 - power(1-x, lambda_gn))/lambda_gn), gamma_gn)
    return Qn
def asymptotic_p_value_Hoeffding(x, n):
    sigma_gn, lambda_gn, gamma_gn, mu_gn = generalized_Weibull_params(n)
    Fn = cdf_generalized_Weibull(sigma_gn, lambda_gn, gamma_gn, mu_gn)
    F = Fn(x)
    p_value = 2*min(F, 1-F)
    return p_value
def Hoeffding_independece_test(option='test'):
    if option == 'Dn':
        return Hoeffding_Dn
    if option == 'nDn + 1/36':
        return lambda X, Y: len(X)*Hoeffding_Dn(X, Y) + 1/36
    if option == 'p value':
        return lambda X, Y: asymptotic_p_value_Hoeffding(len(X)*Hoeffding_Dn(X, Y), len(X))
    else:
        def test(X,Y):
            Dn = Hoeffding_Dn(X, Y)
            p_value = asymptotic_p_value_Hoeffding(len(X) * Dn, len(X))
            return Dn, p_value
        return test