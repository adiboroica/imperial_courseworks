import sys

import numpy as np
import matplotlib.pyplot as plt
import math

NUMBER_OF_SAMPLES = 10000
DEFAULT_REJECTION_REGION = 0.95


def quantile_function(x: float, lambda_value: float) -> float:
    assert (0 < x < 1)
    assert (1 < lambda_value)

    return (1 - x) ** ((1 - lambda_value) ** (-1))


def generate_x_sample(sample_size: int, true_lambda_value: float) -> [float]:
    uniform_sample: [float] = np.random.rand(sample_size)
    x_sample: [float] = list(map(lambda x: quantile_function(x=x, lambda_value=true_lambda_value), uniform_sample))
    return x_sample


def mle(x_sample: [float]):
    sample_size: int = len(x_sample)
    sum_of_logs: float = 0
    for x in x_sample:
        sum_of_logs += math.log(x)
    return 1 + sample_size / sum_of_logs


def generate_mle_array(sample_size: int, true_lambda_value: float) -> [float]:
    mle_array: [float] = []
    for i in range(0, NUMBER_OF_SAMPLES):
        mle_array.append(mle(generate_x_sample(sample_size, true_lambda_value)))
    return mle_array


def right_rejection_region(mle_array: [float], rejection_region: float) -> float:
    assert (0 <= rejection_region <= 1)

    sorted_mle_array: [float] = np.sort(mle_array)
    # index will satisfy the relation
    # ECDF( sorted_mle_array[index - 1] ) < rejection_region <= ECDF( sorted_mle_array[index] )
    index: int = math.ceil(rejection_region * len(sorted_mle_array)) - 1
    return sorted_mle_array[index]


def p_value(mle_array: [float], mle_observation: float):
    sorted_mle_array: [float] = np.sort(mle_array)
    # index satisfies the relation
    # sorted_mle_array[index - 1] < mle_observation <= sorted_mle_array[index]
    index: int = np.searchsorted(sorted_mle_array, mle_observation, side='left')
    return (len(mle_array) - index) / len(mle_array)


def question3_d():
    bins_10_2: [float] = np.linspace(1, 10, 90 + 1)
    bins_10_4: [float] = np.linspace(1, 10, 45 + 1)
    bins_50_4: [float] = np.linspace(1, 10, 90 + 1)

    plt.figure("Question 3 (d)")

    # Add title and axis names
    plt.title("Density Histograms of samples")
    plt.xlabel("MLE of $\lambda$")
    plt.ylabel("Density of samples")

    plt.axvline(2, color='r', label="$\lambda = 2$")
    plt.axvline(4, color='m', label="$\lambda = 4$")

    plt.hist(mle_array_10_2, bins=bins_10_2, density=True, histtype='step', color='b', label="$(n,\lambda) = (10,2)$")
    plt.hist(mle_array_10_4, bins=bins_10_4, density=True, histtype='step', color='g', label="$(n,\lambda) = (10,4)$")
    plt.hist(mle_array_50_4, bins=bins_50_4, density=True, histtype='step', color='y', label="$(n,\lambda) = (50,4)$")

    plt.legend(loc='upper right', prop={'size': 10})

    plt.savefig("build_python/question3_d.png", dpi=600)
    plt.show()


def question3_e_i():
    print("Question 3 (e) i. :")
    print("    Rejection region with Type I error 5%: (", rejection_region_10_2, ", + infinity )")
    print("")


def question3_e_ii():
    sorted_mle_array_10_4: [float] = np.sort(mle_array_10_4)
    # index will satisfy
    # sorted_mle_array_10_4[index - 1] < rejection_region_10_2 <= sorted_mle_array_10_4[index]
    index = np.searchsorted(sorted_mle_array_10_4, rejection_region_10_2, side='left')
    power_of_the_test: float = (len(mle_array_10_4) - index) / len(mle_array_10_4)

    print("Question 3 (e) ii. :")
    print("    The power of the test is: ", power_of_the_test)
    print("")


def question3_e_iii():
    observation: [float] = [1.00, 1.06, 15.69, 1.09, 4.04, 2.20, 2.28, 1.10, 1.46, 1.47]
    mle_observation: float = mle(observation)
    p_value_observation = p_value(mle_array_10_2, mle_observation)

    print("Question 3 (e) iii. :")
    print("    The MLE of this observation is: ", mle_observation)
    if mle_observation < rejection_region_10_2:
        print("    Since the MLE is not in the rejection region, there is insufficient evidence to reject "
              "null hypothesis at the 5% level")
    else:
        print("    Since the MLE is in the rejection region, there is sufficient evidence to reject "
              "the null hypothesis at the 5% level")
    print("    The p-value of this observation is: ", p_value_observation, "(that is,",
          p_value(mle_array_10_2, mle_observation) * 100, "%)")
    print("")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mle_array_10_2: [float] = generate_mle_array(10, 2)
    mle_array_10_4: [float] = generate_mle_array(10, 4)
    mle_array_50_4: [float] = generate_mle_array(50, 4)

    rejection_region_10_2: float = right_rejection_region(mle_array_10_2, DEFAULT_REJECTION_REGION)

    question3_d()

    sys.stdout = open('build_python/question3.txt', "w")
    question3_e_i()
    question3_e_ii()
    question3_e_iii()
