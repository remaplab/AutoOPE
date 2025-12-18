import sys

import numpy as np

SMALL_NUMBER = sys.float_info.min


def total_variance_dist(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total variation distance
    """
    val = np.sum(np.abs(np.subtract(p, q)), axis=1) / 2
    val = np.average(val)
    return val


def pearson_chi_squared_dist(p: np.ndarray, q: np.ndarray):
    val = np.sum(divide(np.power(np.subtract(p, q), 2), q), axis=1)
    val = np.average(val)
    return val


def inner_product_dist(p: np.ndarray, q: np.ndarray):
    val = np.sum(np.multiply(p, q), axis=1)
    val = np.average(val)
    return val


def chebyshev_dist(p: np.ndarray, q: np.ndarray):
    """
    Chebyshev or Total variation distance: formulation with maximum
    """
    val = np.max(np.abs(np.subtract(p, q)), axis=1)
    val = np.average(val)
    return val


def neyman_chi_squared_dist(p: np.ndarray, q: np.ndarray):
    val = np.sum(divide(np.power(np.subtract(p, q), 2), p), axis=1)
    val = np.average(val)
    return val


def div(p: np.ndarray, q: np.ndarray):
    val = 2 * np.sum(divide(np.power(np.subtract(p, q), 2), np.power(np.add(p, q), 2)), axis=1)
    val = np.average(val)
    return val


def canberra_dist(p: np.ndarray, q: np.ndarray):
    val = 2 * np.sum(divide(np.abs(np.subtract(p, q)), np.add(p, q)), axis=1)
    val = np.average(val)
    return val


def k_div(p: np.ndarray, q: np.ndarray):
    val = np.sum(multiply_x_logy(p, np.divide(2 * p, np.add(p, q))), axis=1)
    val = np.average(val)
    return val


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback - Leibler divergence

    p * np.log(p / q) == np.inf when np.log(p / q) == np.inf
    p * np.log(p / q) == np.na when np.log(p / q) == np.inf and p == 0

    Attention: p * np.log(p / q) != np.nan ALWAYS (since p and q are always different from np.nan, and p / q belongs
    to the interval (0, +np.inf] ALWAYS, it cannot be  p / q == np.nan (since p and q belong to [0, 1] ALWAYS)
    """
    val = np.sum(multiply_x_logy(p, divide(p, q)), axis=1)
    val = np.average(val)
    return val


def jensen_shannon_dist(p: np.ndarray, q: np.ndarray):
    val = 0.5 * (np.sum(multiply_x_logy(p, np.divide(2 * p, np.add(p, q))), axis=1) + np.sum(
        multiply_x_logy(q, np.divide(2 * q, np.add(p, q))), axis=1))
    val = np.average(val)
    return val


def kumar_johnson_dist(p: np.ndarray, q: np.ndarray):
    val = np.sum(divide(np.power(np.subtract(np.power(p, 2), np.power(q, 2)), 2), 2 * np.power(np.multiply(p, q),
                                                                                               3 / 2)), axis=1)
    val = np.average(val)
    return val


def additive_symmetric_chi_squared_dist(p: np.ndarray, q: np.ndarray):
    val = np.sum(divide(np.multiply(np.power(np.subtract(p, q), 2), np.add(p, q)), np.multiply(p, q)), axis=1)
    val = np.average(val)
    return val


def euclidian_dist(p: np.ndarray, q: np.ndarray):
    val = np.sqrt(np.sum(np.power(np.subtract(p, q), 2), axis=1))
    val = np.average(val)
    return val


def kulczynski_dist(p: np.ndarray, q: np.ndarray):
    val = divide(np.sum(np.abs(np.subtract(p, q)), axis=1), np.sum(np.minimum(p, q), axis=1))
    val = np.average(val)
    return val


def city_block(p: np.ndarray, q: np.ndarray):
    val = np.sum(np.abs(np.subtract(p, q)), axis=1)
    val = np.average(val)
    return val


def multiply_x_logy(x, y):
    log_y = np.log(y)

    x_zero = x == 0
    log_zero = log_y == 0
    zero_log_zero = np.logical_and(x_zero, log_zero)

    product = np.multiply(x, log_y)
    product[zero_log_zero] = 0

    return product


def divide(x, y):
    quotient = np.divide(x, y)

    x_zero = x == 0
    div_by_zero = y == 0
    zero_by_zero = np.logical_and(x_zero, div_by_zero)

    quotient[div_by_zero] = SMALL_NUMBER
    quotient[zero_by_zero] = 0

    return quotient
