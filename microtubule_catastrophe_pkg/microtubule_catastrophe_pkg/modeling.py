import numpy as np
import pandas as pd
import scipy
import scipy.stats as st

import warnings
import tqdm

import bebi103 

"""
Functions for modeling microtubule catastrophe data
"""

def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. gamma measurements."""
    alpha, beta = params

    # constraints on number of steps and rate
    if alpha <= 0 or beta <= 0:
        return -np.inf
    
    return np.sum(st.gamma.logpdf(n, alpha, loc=0, scale=1/beta))

def mle_iid_gamma(n, x0):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma variables"""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array(x0),
            args=(n,),
            method='Powell'
        )

    if res.success:
         return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def gen_gamma(alpha, beta, x0, size):
    '''Generates samples from the model gamma distribution'''
    return np.random.gamma(alpha, 1/beta, size=size)

def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args), *args) for _ in iterator]
    )

def log_likelihood_model(n, beta1, beta2): 
    """Computes log likelihood for measurements from the derived model
    contained in n, with parameters beta1 and beta2."""
    
    dbeta = beta2 - beta1
    
    l = []
    
    # compute log likelihood at each time
    for t in n:
        L = beta1*(dbeta + beta1) / dbeta * np.exp(-beta1*t) * (1 - np.exp(-dbeta*t))
        l.append(np.log(L))
        
    return l

def log_like_iid_model(params, n):
    """Log likelihood for i.i.d. measurements from the derived model."""
    beta1, beta2 = params

    # constraints on parameters
    if beta1 >= beta2 or beta1 <= 0:
        return -np.inf
    
    return np.sum(log_likelihood_model(n, beta1, beta2))

def mle_iid_model(n, x0):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    variables from modeled distribution"""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_model(params, n),
            x0=np.array(x0),
            args=(n,),
            method='Powell'
        )

    if res.success:
         return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def gen_model(beta1, beta2, x0, size):
    '''Generates samples from the model distribution'''
    
    x1 = np.random.exponential(1/beta1, size)
    x2 = np.random.exponential(1/beta2, size)
    
    return x1 + x2
        
def theor_cdf_custom(beta1, beta2, t):
    '''Computes theoretical CDF of custom model at times t
    given parameters beta1, beta2'''
    
    first_term = beta1 * beta2 / (beta2 - beta1)
    second_term = (1 / beta1 * (1 - np.exp(-beta1 * t)) - 1 / beta2 * (1 - np.exp(-beta2 * t)))
    
    return first_term * second_term

def compute_AIC(data, x0_gamma, x0_custom):
    '''Computes AIC and Akaike weights for model comparison between 
    Gamma and custom models
    
    Arguments:
    data: observed measurements of time to catastrophe
    x0_gamma: initial guesses for optimization for Gamma MLEs
    x0_custom: initial guesses for optimization for Custom MLEs
    
    Returns:
    df_mle: dataframe containing MLEs, AIC and loglikelihood values, and 
    Akaike weights for both models
    '''
    
    # Gamma MLE
    alpha, beta = mle_iid_gamma(data, x0_gamma)

    # Custom MLE
    beta1, beta2 = mle_iid_model(data, x0_custom)

    # Store results in data frame
    df_mle = pd.DataFrame([alpha, beta, beta1, beta2], index=['alpha', 'beta', 'beta1', 'beta2'], columns=['parameters'])
    
    # Gamma model
    gamma_time_to_catas = log_like_iid_gamma(df_mle.loc[["alpha", "beta"]].values, data)
    df_mle.loc["log_like_gamma"] = gamma_time_to_catas

    # Custom model
    model_time_to_catas = log_like_iid_model(
        df_mle.loc[["beta1", 'beta2']].values,
        data,
    )
    df_mle.loc["log_like_custom"] = model_time_to_catas
    
    df_mle.loc['AIC_gamma'] = -2 * (df_mle.loc['log_like_gamma'] - 2)
    df_mle.loc['AIC_custom'] = -2 * (df_mle.loc['log_like_custom'] - 2)

    # weight for Gamma model
    AIC_max = max(df_mle.loc[['AIC_gamma', 'AIC_custom']]['parameters'])
    numerator = np.exp(-(df_mle.loc['AIC_gamma'] - AIC_max)/2)
    denominator = numerator + np.exp(-(df_mle.loc['AIC_custom'] - AIC_max)/2)
    df_mle.loc['w_gamma'] = numerator / denominator

    # weight for custom model (weights should add to 1)
    numerator = np.exp(-(df_mle.loc['AIC_custom'] - AIC_max)/2)
    df_mle.loc['w_custom'] = numerator / denominator

    return df_mle