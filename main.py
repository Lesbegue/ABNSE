import numpy as np
import matplotlib.pyplot as plt
from abnse import *
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
method_ = 'PBNSE'
Data_name = ['hr2']
global wq, vq, theta, sigema_n
u = -1
while u < 0:
    u = u + 1
    v = -1
    mean_psd_list = []
    while v < len(Data_name) - 1:
        v = v + 1
        mean_CBNSE = -1
        example = Data_name[v]
        if example == 'hr1':
            signal_raw = np.loadtxt('data/hr1.txt')
            time = np.linspace(0, len(signal_raw), len(signal_raw))
            signal_label = 'heart-rate(HR1)'
            time_label = 'time'
            size = 1 / 3
            m = 4

        elif example == 'hr2':
            signal_raw = np.loadtxt('data/hr2.txt')
            time = np.linspace(0, len(signal_raw), len(signal_raw))
            signal_label = 'heart-rate signal(HR2)'
            time_label = 'time'
            size = 1 / 3
            m = 3

        elif example == 'sunspots':
            dta = sm.datasets.sunspots.load_pandas().data
            signal_raw = np.array(dta.SUNACTIVITY)
            time = np.array(dta.YEAR)
            time_label = 'time'
            signal_label = 'sunspot_data'
            size = 1 / 3
            m = 3

        elif example == 'Eye State':
            signal_raw = np.loadtxt('data/eye_state.txt')
            signal_raw = signal_raw[:1800]
            time = np.linspace(0, len(signal_raw), len(signal_raw))
            time_label = 'time'
            signal_label = 'Eye State_data'
            size = 1 / 3
            m = 3

        elif example == 'Bike Sharing':
            signal_raw = np.loadtxt('data/bike sharing.txt')
            # signal_raw = signal_raw[:5400]
            time = np.linspace(0, len(signal_raw), len(signal_raw))
            time_label = 'time'
            signal_label = 'Bike Sharing_data'
            size = 1 / 10
            m = 3

        elif example == 'sin_cos':
            signal_raw = np.loadtxt('data/sin_cos_time_dependent_heteroscedastic.txt')
            time = np.linspace(0, len(signal_raw), len(signal_raw))
            time_label = 'time'
            signal_label = 'sin_cos'
            size = 1 / 3
            m = 6

        # Centralize the signal and time
        signal_raw = signal_raw - np.mean(signal_raw)
        time = time - np.mean(time)

        # determine the partially random sampled signal.
        indices = np.random.randint(0, len(signal_raw), int(len(signal_raw) * size))
        signal_ind = signal_raw[indices]
        time_ind = time[indices]

        # Define the slicing of signals, tao is a list of the center of each segment.
        semi_window_len = len(signal_raw) / 2 / m
        if m % 2 == 0:
            tao = [(2 * int(i) + 1) * semi_window_len for i in np.linspace(-m / 2, m / 2 - 1, m)]
        else:
            tao = [(2 * int(i)) * semi_window_len for i in np.linspace(-(m - 1) / 2, (m - 1) / 2, m)]


        # Define the radius of each segment.
        b = 2 * semi_window_len

        # Initiate the posteriors of the mean and covariance of the full signal.
        post_mean_r_full = 0
        post_mean_i_full = 0
        post_cov_r_full = 0
        post_cov_i_full = 0
        post_cov_F_full = 0

        # Loop over each segment and train the signal GP model.
        fig_time, ax_time = plt.subplots(figsize=(16, 9))
        fig_real, ax_real = plt.subplots(figsize=(16, 9))
        fig_imag, ax_imag = plt.subplots(figsize=(16, 9))
        if method_ == 'PBNSE':
            for i in range(m):
                index = [time_ind.tolist().index(j) for j in time_ind if tao[i] - b / 2-1 <= j <= tao[i] + b / 2] # find the index of each segment
                if i == 0:
                    my_bse = bse(time_ind[index], signal_ind[index], tao[i], b)  # the signal GP model for the first segment
                    my_bse.set_labels(time_label, signal_label)
                    if example == 'hr1':
                        my_bse.set_freqspace(0.03)
                    elif example == 'hr2':
                        my_bse.set_freqspace(0.03)
                    elif example == 'sunspots':
                        my_bse.set_freqspace(0.2)
                    elif example == 'sin_cos':
                        my_bse.set_freqspace(0.02)
                    elif example == 'Eye State':
                        my_bse.set_freqspace(0.03)
                    elif example == 'Bike Sharing':
                        my_bse.set_freqspace(0.001)
                    nll = my_bse.neg_log_likelihood()
                    print(f'Negative log likelihood (before training): {nll}')
                    my_bse.compute_moments()

                    [wq, vq, theta, sigma_n] = my_bse.train()  # train the signal GP model for the first segment and get the parameters.
                    nll = my_bse.neg_log_likelihood()
                    print(f'Negative log likelihood (after training): {nll}')

                    # compute the moments of the signal GP model for the first segment.
                    w, post_mean, post_cov, post_mean_r, post_cov_r, post_mean_i, post_cov_i , post_cov_F= my_bse.compute_moments()

                    # add the mean and covariance of this segment to the full signal
                    post_mean_r_full = post_mean_r
                    post_mean_i_full = post_mean_i
                    post_cov_r_full = post_cov_r
                    post_cov_i_full = post_cov_i
                    post_cov_F_full = post_cov_F

                    print("==================", "The", i, "window", "=============================")

                    fig_time, ax_time = my_bse.plot_interpolation(flag=None, loc=None, fig=fig_time, ax=ax_time)
                    # plot the PSD of the first segment.
                    fig, ax = plt.subplots(figsize=(16, 9))
                    w, posterior_mean_psd, ax, fig = my_bse.plot_power_spectral_density_new(15, flag=None,n=i+1,fig=fig,ax=ax)
                    fig.savefig(r'Fitting, spectrum, and PSD of signals\PSD_of {}-th part.pdf'.format(i + 1))
                else:
                    my_bse = bse(time_ind[index], signal_ind[index], tao[i], b)  # the signal GP model for the downstream segments
                    my_bse.set_labels(time_label, signal_label)
                    if example == 'hr1':
                        my_bse.set_freqspace(0.03)
                    elif example == 'hr2':
                        my_bse.set_freqspace(0.03)
                    elif example == 'sunspots':
                        my_bse.set_freqspace(0.2)
                    elif example == 'sin_cos':
                        my_bse.set_freqspace(0.02)
                    elif example == 'Eye State':
                        my_bse.set_freqspace(0.03)
                    elif example == 'Bike Sharing':
                        my_bse.set_freqspace(0.001)

                    # Transfering the parameters of the former segment to this signal GP.
                    my_bse.Assign(wq, vq, theta, sigma_n, tao[i], b)
                    # train the signal GP model and get the parameters.
                    [wq, vq, theta, sigma_n] = my_bse.train()

                    nll = my_bse.neg_log_likelihood()
                    print(f'Negative log likelihood (after training): {nll}')
                    w, post_mean, post_cov, post_mean_r, post_cov_r, post_mean_i, post_cov_i ,post_cov_F = my_bse.compute_moments()

                    # add the mean and covariance of this segment to the full signal
                    post_mean_r_full = post_mean_r_full + post_mean_r
                    post_mean_i_full = post_mean_i_full + post_mean_i
                    post_cov_r_full = post_cov_r_full + post_cov_r
                    post_cov_i_full = post_cov_i_full + post_cov_i
                    post_cov_F_full = post_cov_F_full + post_cov_F
                    print("==================", "The", i, "window", "=============================")

                    fig_time, ax_time = my_bse.plot_interpolation(flag=None, loc=None, fig=fig_time, ax=ax_time)
                    # plot the PSD of this segment.
                    fig, ax = plt.subplots(figsize=(16, 9))
                    w, posterior_mean_psd, ax, fig = my_bse.plot_power_spectral_density_new(15, flag=None, n=i+1,fig = fig,ax = ax)
                    fig.savefig(
                        r'Fitting, spectrum, and PSD of signals\PSD_of {}-th part.pdf'.format(i+1))
            # plot the PSD of the full signal.
            ax_time.set_title('Observations and posterior interpolation')
            ax_time.set_xlabel(time_label)
            handles, labels = ax_time.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_time.legend(by_label.values(), by_label.keys())
            ax_time.set_xlim([min(time_ind), max(time_ind)])
            fig_time.tight_layout()
            fig_time.savefig(
                r'Fitting, spectrum, and PSD of signals\interpolation_of full signal.pdf')

            ax_real.plot(w, post_mean_r_full, color='blue', label='posterior mean')
            error_bars = 2 * np.sqrt((np.diag(post_cov_r_full)))
            ax_real.fill_between(w, post_mean_r_full - error_bars, post_mean_r_full + error_bars, color='blue',
                              alpha=0.1, label='95% error bars')
            ax_real.set_title('Posterior spectrum (real part)')
            ax_real.legend()
            ax_real.set_xlabel(r'frequency $ \omega $')
            ax_real.set_xlim([min(w), max(w)])
            fig_real.tight_layout()
            fig_real.savefig(r"Fitting, spectrum, and PSD of signals\Posterior spectrum real part.pdf", pad_inches=0)

            ax_imag.plot(w, post_mean_i_full, color='blue', label='posterior mean')
            error_bars = 2 * np.sqrt((np.diag(post_cov_i_full)))
            ax_imag.fill_between(w, post_mean_i_full - error_bars, post_mean_i_full + error_bars, color='blue',
                                 alpha=0.1, label='95% error bars')
            ax_imag.set_title('Posterior spectrum (imaginary part)')
            ax_imag.legend()
            ax_imag.set_xlabel(r'frequency $ \omega $')
            ax_imag.set_xlim([min(w), max(w)])
            fig_imag.tight_layout()
            fig_imag.savefig(r"Fitting, spectrum, and PSD of signals\Posterior spectrum imaginary part.pdf", pad_inches=0)


            fig, ax = plt.subplots(figsize=(16, 9))
            post_mean_F_full = np.concatenate((post_mean_r_full, post_mean_i_full))
            color_list = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
            freqs = len(w)
            how_many = 15
            samples = np.zeros((freqs, how_many))
            for i in range(how_many):
                sample = np.random.multivariate_normal(post_mean_F_full,
                                                       (post_cov_F_full + post_cov_F_full.T) / 2 + 1e-5 * np.eye(
                                                           2 * freqs))
                samples[:, i] = sample[0:freqs] ** 2 + sample[freqs:] ** 2
            ax.plot(w, samples, linewidth=0.1, color='red', alpha=0.3)
            ax.plot(w, samples[:, 0], color='red', linewidth=0.1, alpha=0.3, label='posterior samples of full signal')

            posterior_mean_psd = post_mean_r_full ** 2 + post_mean_i_full ** 2 + np.diag(
                post_cov_r_full + post_cov_i_full)

            ax.plot(w, posterior_mean_psd, color='black', linewidth=2.0,
                    label='(analytical) posterior mean of full signal')
            ax.set_title('Sample posterior power spectral density')
            ax.set_xlabel(r'frequency $\omega$')
            ax.legend()
            ax.set_xlim([min(w), max(w)])
            fig.savefig(r'Fitting, spectrum, and PSD of signals\PSD_of full signal .pdf'.format(i + 1))
            plt.close('all')

