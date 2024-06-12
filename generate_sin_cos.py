import numpy as np
import matplotlib.pyplot as plt

# Set plotting parameters
plot_params = {
    'legend.fontsize': 26,
    'figure.figsize': (16, 9),
    'xtick.labelsize': '18',
    'ytick.labelsize': '18',
    'axes.titlesize': '24',
    'axes.labelsize': '22'
}
plt.rcParams.update(plot_params)

def f(t):
    return 10 * np.cos(2 * np.pi * 0.5 * t)- 5 * np.sin(2 * np.pi * 1 * t)

def generate_data_with_time_dependent_heteroscedastic_noise(num_points=1800):
    t = np.linspace(-10, 10, num_points)
    noise_variances = 0.1 + 1 * np.sin(2 * np.pi * t / 10)  # Time-dependent noise variance function
    noise_variances = np.maximum(noise_variances, 0)  # Ensure all noise variances are greater than or equal to 0
    y = f(t) + np.random.normal(0, np.sqrt(noise_variances), size=num_points)  # Add time-dependent noise
    return t, y

def plot_time_series(t, y):
    plt.plot(t, y, label='f(t) with Time-Dependent Heteroscedastic Noise')
    plt.plot(t, f(t), 'r--', label='f(t) without Noise')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Sampled Time Series with Time-Dependent Heteroscedastic Noise')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate data and plot
t, y = generate_data_with_time_dependent_heteroscedastic_noise()
np.savetxt('data/sin_cos_time_dependent_heteroscedastic.txt', y, delimiter='', newline='\n')
plot_time_series(t, y)
