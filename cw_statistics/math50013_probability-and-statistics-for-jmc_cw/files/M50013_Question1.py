import math
import sys

import numpy as np
import scipy.stats as st
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_SAMPLE_SIZE: int = 20
DEFAULT_CI_VALUE: float = 0.95
TRUE_VALUE_OF_INTEGRAL: float = 1.81280
NUMBER_OF_EXPERIMENTS: int = 100


def g(x: float):
    return math.exp(-1 * x * x * x * x)


def f(x: float, sigma: float = 1):
    return st.norm.pdf(x=x, scale=sigma)


def get_random_weights(number_of_samples: int, sigma: float = 1) -> [float]:
    assert (sigma > 0)
    assert (number_of_samples > 0)

    x_normal_sample: [float] = sigma * np.random.randn(number_of_samples)
    return list(map(lambda x: g(x) / f(x, sigma=sigma), x_normal_sample))


def estimation_of_integral(weights: [float]) -> ndarray:
    return np.mean(weights)


def confidence_interval(alpha: float, weights: [float]) -> [float]:
    return st.t.interval(alpha=alpha, df=len(weights) - 1, loc=estimation_of_integral(weights), scale=st.sem(weights))


def estimate_integral_and_confidence_interval(number_of_samples: int, sigma: float = 1):
    assert (0 < number_of_samples)

    sample: [float] = get_random_weights(number_of_samples, sigma)

    print("Estimation of the given integral using", number_of_samples, "samples :")
    print("    estimate of the integral: ", estimation_of_integral(sample))
    print("    95% confidence interval: ", confidence_interval(DEFAULT_CI_VALUE, sample))
    print("")


def std_of_100_experiments(number_of_samples: int) -> ndarray:
    estimates: [float] = np.empty(NUMBER_OF_EXPERIMENTS, dtype=float)
    for i in range(0, NUMBER_OF_EXPERIMENTS):
        estimates[i] = estimation_of_integral(get_random_weights(number_of_samples))
    return np.std(estimates)


def question1_d():
    sample: [float] = get_random_weights(DEFAULT_SAMPLE_SIZE)

    print("Question 1 (d) :")
    print("    estimate of the integral: ", estimation_of_integral(sample))
    print("    95% confidence interval: ", confidence_interval(DEFAULT_CI_VALUE, sample))
    print("")


def question1_e():
    sigma_values: [float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    plt.figure("Question 1 (e)")

    plt.title("Estimations and Confidence Intervals for the integral I")
    plt.xlabel("$\sigma$ values")
    plt.ylabel("Estimations of the integral I")

    plt.ylim(-1, 7)

    plt.axhline(TRUE_VALUE_OF_INTEGRAL, color='red', label="true value of the integral")

    for sigma in sigma_values:
        weights: [float] = get_random_weights(number_of_samples=DEFAULT_SAMPLE_SIZE, sigma=sigma)
        estimate: ndarray = estimation_of_integral(weights=weights)
        ci_interval: [float] = confidence_interval(DEFAULT_CI_VALUE, weights=weights)

        plt.plot([sigma, sigma], ci_interval, color='c')
        plt.plot(sigma, estimate, marker='o', color='k')

    handles, labels = plt.gca().get_legend_handles_labels()

    estimation_label = Line2D([0], [0], marker='o', label='estimation', color='k', lw=0)
    ci_label = Line2D([0], [0], label='confidence interval', color='c')

    handles.extend([estimation_label, ci_label])

    plt.legend(loc='upper right', handles=handles, prop={'size': 10})

    plt.savefig("build_python/question1_e.png", dpi=600)
    plt.show()


def question1_f():
    # TODO: Improve the complexity of this algorithm. (it takes 50 seconds to run)

    n: int = 2 ** 3
    sigma: ndarray = std_of_100_experiments(n)

    plt.figure("Question 1 (f)")

    plt.title("log-log plot of $n$ and average standard deviation")
    plt.xlabel("$log_2 (n)$")
    plt.ylabel("Average $log_2$ standard deviation of samples")

    plt.loglog(n, sigma, 'o', color='k',
               base=2)
    plt.loglog([2 ** 3, 2 ** 12], [sigma, sigma * (1 / math.sqrt(2 ** 9))], color='r', label="predicted scaling",
               base=2)

    for i in range(4, 12 + 1):
        n = n * 2
        std = std_of_100_experiments(n)
        plt.loglog(n, std, 'o', color='k',
                   base=2)

    handles, labels = plt.gca().get_legend_handles_labels()

    standard_deviation_label = Line2D([0], [0], marker='o', label='average standard deviation', color='k', lw=0)

    handles.extend([standard_deviation_label])

    plt.legend(loc='upper right', handles=handles, prop={'size': 10})

    plt.savefig("build_python/question1_f.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    sys.stdout = open('build_python/question1.txt', "w")
    question1_d()
    question1_e()
    question1_f()
