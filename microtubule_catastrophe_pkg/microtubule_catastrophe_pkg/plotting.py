import iqplot
import bebi103 

import bokeh.io

from modeling import theor_cdf_custom

bokeh.io.output_notebook()
bebi103.hv.set_defaults()

"""
Functions to plot microtubule catastrophe data
"""

def ecdf_labeled_unlabeled(df):
    '''Generates ECDF plot for times to catastrophe
    for labeled and unlabeled tubulin
    
    Website Figure 1
    '''
    p = iqplot.ecdf(
        data=df,
        q='time to catastrophe (s)',
        cats='labeled',
        conf_int=True,
        x_axis_label='time (sec)',
        y_axis_label='empirical ecdf',
        title='ECDF of Time to Catastrophe'
    )
    
    return p

def ecdf_beta_ratios_df(beta1, beta2):
    '''Generates dataframe for plotting ECDF of Times to Catastrophe 
    according to custom model parametrized by various rates
    
    Arguments: 
    beta1: array of rates for first Poisson process
    beta2: array of rates for second Poisson process
    '''

    # compute ratios between beta2 and beta1
    ratio = []
    for j in range(len(beta1)):
        ratio.append(beta2[j] / beta1[j])

    # creating column for beta2/beta1 ratios to use in DataFrame
    ratio_col = []
    for i in ratio:
        ratio_col.append([i]*150)

    ratio_col = np.array(ratio_col).flatten()

    # create column of time values
    val_col = []
    for i in range(len(beta1)):
        x1 = np.random.exponential(1/beta1[i], 150)
        x2 = np.random.exponential(1/beta2[i], 150)
        val_col.append(x1 + x2)

    val_col = np.array(val_col).flatten()

    # create the dataframe
    df = pd.DataFrame({'beta2/beta1 ratio': ratio_col, 'total time (1/beta1)': val_col})
    
    return df

def ecdf_beta_ratios(df):
    '''Plots ECDF of Times to Catastrophe according to 
    custom model parametrized by various rates
    
    Website Figure 2
    '''
    p = iqplot.ecdf(
        data=df,
        q = 'total time (1/beta1)',
        cats = ['beta2/beta1 ratio'],
        style='staircase',
        title='Time to Catastrophe for Different Beta2/Beta1'
    )

    p.legend.title = 'beta2/beta1'

    return p
    
def ecdf_vs_theor_cdf(beta1, beta2):
    '''Plots theoretical CDF vs simulated ECDF of custom model
    given parameters beta1, beta2
    
    Website Figure 3
    '''
    
    # simulated ECDF values
    df = ecdf_beta_ratios_df([beta1], [beta2])

    p = iqplot.ecdf(data=df,
            q = 'total time (1/beta1)',
            cats = ['beta2/beta1 ratio'],
            show_legend=False,
            title='Analytical CDF vs Simulated ECDF for Time to Catastrophe'
        )

    # plot analytical CDF
    t = np.linspace(0, max(df['total time (1/beta1)']))
    f = theor_cdf_custom(beta1, beta2)

    p.line(x=t, y=f, line_width=2, line_color='red')

    # add a legend
    legend = bokeh.models.Legend(
                items=[('analytical CDF', [p.circle(color='red')]),
                       ('ECDF', [p.circle(color='blue')])
                      ],
                location='center')
    p.add_layout(legend, 'right')

    return p

def qq_plot(df, samples, model_name):
    '''Generate QQ plot of generative vs observed quantiles for 
    microtubule catastrophe data
    
    Arguments:
    df: dataframe of experimental measurements
    samples: samples drawn from generative distribution
    model_name: string, either 'Gamma' or 'Custom'
    
    Website Figures 5 and 6
    '''
    p = bebi103.viz.qqplot(
        data=df,
        samples=samples,
        x_axis_label="time to catastrophe (s)",
        y_axis_label="time to catastrophe (s)",
        title='Q-Q Plot for {} Model'.format(model_name)
    )

    return p

def predictive_ecdf(data, samples, model_name, diff=False):
    '''
    Generates predictive ECDFs to compare between the generative
    distribution and measured data.
    
    Arguments:
    data: measured data
    samples: generative samples
    model_name: string, either 'Gamma' or 'Custom'
    diff: whether to compute ECDF differences or not
    
    Website Figures 7-10
    '''
    
    if diff:
        p = bebi103.viz.predictive_ecdf(
            samples=samples, 
            data=data, 
            x_axis_label='time to catastrophe (s)', 
            title='Predictive ECDFs for {} Model'.format(model_name),
            diff='ecdf'
        )
    else:
        p = bebi103.viz.predictive_ecdf(
            samples=samples, 
            data=data, 
            x_axis_label='time to catastrophe (s)', 
            title='Predictive ECDFs for {} Model'.format(model_name)
        )
    
    return p

def plot_conf_ints(estimates, conf_ints, labels):
    '''
    Visualization of confidence intervals 
    
    Arguments:
    estimates: MLEs of parameter of interest
    conf_ints: confidence intervals for MLEs in estimates 
    labels: labels for each confidence interval to be plotted
    
    Website Figures 4, 11, 12
    '''
    
    summaries = [
        dict(estimate=est, conf_int=conf, label=name)
        for est, conf, name in zip(
            estimates, conf_ints, labels
        )
    ]

    p = bebi103.viz.confints(summaries)

    return p
