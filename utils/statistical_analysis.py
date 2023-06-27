# CEEMDAN_LSTM包里面用于判断序列统计特性的函数，例如自相关性
import pandas as pd
import matplotlib.pyplot as plt


# 检测传入的是否是一个pd.Series
def check_series(series):
    try:
        series = pd.Series(series)
    except:
        raise ValueError(
            'Sorry! %s is not supported for the Hybrid Method, please input pd.DataFrame, pd.Series, nd.array(<=2D)' % type(
                series))
    return series


# 4.Main. Statistical tests
def statis_tests(series=None):
    """
    Make statistical tests, including ADF test, Ljung-Box Test, Jarque-Bera Test, and plot ACF and PACF, to evaluate stationarity, autocorrelation, and normality.
    Input: series     - the time series (1D)
    """
    adf_test(series)  # Dickey-Fuller test (ADF test)   衡量平稳性
    LB_test(series)   # Ljung-Box Test                  衡量自相关性
    jb_test(series)   # Jarque-Bera Test                是否为正态分布
    plot_acf_pacf(series)  # 绘图


# 4.1 ADF test
def adf_test(series=None):
    """
    Make Augmented Dickey-Fuller test (ADF test) to evaluate stationary.
    ADF测试是一种常用的统计检验，用于检验给定的时间序列是否平稳
    平稳性是指时间序列的均值、方差和自协方差不随时间变化的性质。平稳性是很多时间序列模型的基本假设，如果时间序列不平稳，就可能存在单位根，即滞后项系数为的自回归过程
    ADF测试的原理是对时间序列进行一阶或高阶差分，然后对差分后的序列进行回归分析，得到一个检验统计量，再与临界值进行比较，判断是否拒绝原假设
    ADF测试的原假设是存在单位根，即时间序列不平稳；备择假设是不存在单位根，即时间序列平稳
    ADF测试可以通过Python中的statsmodels模块进行实现，该模块提供了adfuller函数来执行ADF测试，并返回检验统计量、p值、滞后阶数、观测值数目、临界值等信息
    Input: series     - the time series (1D)
    """
    from statsmodels.tsa.stattools import adfuller  # adf_test
    series = check_series(series)
    adf_ans = adfuller(series)  # The outcomes are test value, p-value, lags, degree of freedom.
    print('==========ADF Test==========')
    print('Test value:', adf_ans[0])
    print('P value:', adf_ans[1])
    print('Lags:', adf_ans[2])
    print('1% confidence interval:', adf_ans[4]['1%'])
    print('5% confidence interval:', adf_ans[4]['5%'])
    print('10% confidence interval:', adf_ans[4]['10%'])
    # print(adf_ans)

    # Brief review
    adf_status = ''
    if adf_ans[0] <= adf_ans[4]['1%']:
        adf_status = 'very strong'
    elif adf_ans[0] <= adf_ans[4]['5%']:
        adf_status = 'strong'
    elif adf_ans[0] <= adf_ans[4]['10%']:
        adf_status = 'normal'
    else:
        adf_status = 'no'
    print('The p-value is ' + str(adf_ans[1]) + ', so the series has ' + str(adf_status) + ' stationarity.')
    print('The automatic selecting lags is ' + str(adf_ans[2]) + ', advising the past ' + str(
        adf_ans[2]) + ' days as the features.')


# 4.2 Ljung-Box Test
def LB_test(series=None):
    """
    Make Ljung-Box Test to evaluate autocorrelation.
    Ljung-Box测试是一种统计检验，用于检验时间序列的一组自相关是否不为零
    它不是在每个不同的滞后处测试随机性，而是基于多个滞后测试“整体”随机性，因此是一个组合测试
    Ljung-Box测试的原理是构造一个Q统计量，用来检测一个时间序列数据所有k阶自相关系数是否联合为零。
    Q统计量服从自由度为k的卡方分布，可以与临界值进行比较，判断是否拒绝原假设
    Ljung-Box测试的原假设是不存在自相关性，即时间序列是白噪声；备择假设是存在自相关性，即时间序列非白噪声
    Input: series     - the time series (1D)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test  # LB_test
    series = check_series(series)  # 检查是否为Series
    lb_ans = lb_test(series, lags=None, boxpierce=False)  # The default lags=40 for long series.
    print('==========Ljung-Box Test==========')  # lb_ans是一个df,形状为: (10,2)
    # pd.Series(lb_ans[1]).plot(title='Ljung-Box Test p-values')  # Plot p-values in a figure
    # if np.sum(lb_ans[1]) <= 0.05:  # Brief review
    if lb_ans.apply(lambda x: x.sum(), axis=0)[1] <= 0.05:  # Brief review
        print('The sum of p-value is ' + str(lb_ans.apply(lambda x: x.sum(), axis=0)[1])
              + '<=0.05, rejecting the null hypothesis that the series has very strong autocorrelation.')
    else:
        print('Please view with the line chart, the autocorrelation of the series may be not strong.')
    # print(pd.DataFrame(lb_ans)) # Show outcomes with test value at line 0, and p-value at line 1.


# 4.3 Jarque-Bera Test
def jb_test(series=None):
    """
    Make Jarque-Bera Test to evaluate normality (whether conforms to a normal distribution).
    Jarque-Bera测试是一种拟合优度检验，用于检验样本数据是否具有与正态分布匹配的偏度和峰度
    Jarque-Bera测试的原理是构造一个JB统计量，用来衡量样本数据的偏度和峰度与正态分布的偏度和峰度之间的差异。
    JB统计量服从自由度为2的卡方分布，可以与临界值进行比较，判断是否拒绝原假设。
    Jarque-Bera测试的原假设是样本数据服从正态分布；备择假设是样本数据不服从正态分布。
    Input: series     - the time series (1D)
    """
    from statsmodels.stats.stattools import jarque_bera as jb_test  # JB_test
    series = check_series(series)
    jb_ans = jb_test(series)  # The outcomes are test value, p-value, skewness and kurtosis.
    print('==========Jarque-Bera Test==========')
    print('Test value:', jb_ans[0])
    print('P value:', jb_ans[1])
    print('Skewness:', jb_ans[2])
    print('Kurtosis:', jb_ans[3])
    # Brief review
    if jb_ans[1] <= 0.05:
        print(
            'p-value is ' + str(jb_ans[1]) + '<=0.05, rejecting the null hypothesis that the series has no normality.')
    else:
        print('p-value is ' + str(
            jb_ans[1]) + '>=0.05, accepting the null hypothesis that the series has certain normality.')


# 4.4 Plot ACF and PACF figures
def plot_acf_pacf(series=None, fig_path=None):
    """
    Plot ACF and PACF figures to evaluate autocorrelation and find the lag.
    Input:
    series     - the time series (1D)
    fig_path   - the figure saving path
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # plot_acf_pacf
    series = check_series(series)
    print('==========ACF and PACF==========')
    fig = plt.figure(figsize=(10, 5))
    fig1 = fig.add_subplot(211)
    plot_acf(series, lags=40, ax=fig1)
    fig2 = fig.add_subplot(212)
    plot_pacf(series, lags=40, ax=fig2)
    # Save the figure
    if fig_path is not None:  # 这里 fig_path 为None,所以默认没有保存图片
        plt.savefig(fig_path + 'Figures_ACF_PACF.jpg', dpi=300, bbox_inches='tight')
        plt.tight_layout()
    plt.show()


# Plot Heatmap
def plot_heatmap(data, corr_method='pearson', fig_path=None):
    """
    Plot heatmap to check the correlation between variables.
    Input:
    data         - the 2D array
    corr_method  - the method to calculate the correlation
    fig_path     - the figure saving path
    """
    try:
        import seaborn as sns
    except:
        raise ImportError('Cannot import seaborn, run: pip install seaborn!')
    try:
        data = pd.DataFrame(data)
    except:
        raise ValueError('Invalid input!')
    f, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(data.corr(corr_method), cmap='OrRd', linewidths=0.05, ax=ax, annot=True, fmt='.5g')  # RdBu
    if fig_path is not None:
        plt.savefig(fig_path + '_Heatmap.jpg', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    plt.show()


# 4.5 DM test
def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    """
    Copyright (c) 2017 John Tsang
    Author: John Tsang https://github.com/johntwk/Diebold-Mariano-Test
    Diebold-Mariano-Test (DM test) is used to compare time series forecasting result performance.

    Input and Parameters:
    ---------------------
    actual_lst    - target values in test set
    pred1_lst     - forecasting result
    pred2_lst     - another forecasting result
    h             - the number of stpes ahead
    crit          - a string specifying the criterion eg. MSE, MAD, MAPE, poly
                        1)  MSE : the mean squared error
                        2)  MAD : the mean absolute deviation
                        3) MAPE : the mean absolute percentage error
                        4) poly : use power function to weigh the errors
    poly          - the power for crit power (it is only meaningful when crit is "poly")

    Output:
    ---------------------
    DM	          - The DM test statistics
    p-value	      - The p-value of DM test statistics
    """
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt, msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt, msg)
        len_act = len(actual_lst)
        len_p1 = len(pred1_lst)
        len_p2 = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt, msg)
        # Check range of h
        if h >= len_act:
            rt = -1
            msg = "The number of steps ahead is too large."
            return rt, msg
        # Check if criterion supported
        if crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly":
            rt = -1
            msg = "The criterion is not supported."
            return (rt, msg)
            # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")

        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True

        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt, msg)
        return (rt, msg)

    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np

    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    # construct d according to crit
    if (crit == "MSE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1) ** 2)
            e2_lst.append((actual - p2) ** 2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1) / actual))
            e2_lst.append(abs((actual - p2) / actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1)) ** (power))
            e2_lst.append(((actual - p2)) ** (power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

            # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
    DM_stat = harvey_adj * DM_stat
    # Find p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM=DM_stat, p_value=p_value)
    return rt
