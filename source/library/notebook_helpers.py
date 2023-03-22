from typing import Callable, Optional
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import arviz as az
from helpsk.string import format_number


def plot_hdi(
        samples,
        title: Optional[str] = None,
        transformation: Optional[Callable] = None,
        vertical_factor_66: int = 1,
        vertical_factor_95: int = 1,
        ticks: Optional[list] = None,
        decimals: int = 1):
    """
    Plot the 66% and 95% Hight Density Interval of `samples` passed in. The min, median, and max
    is also plotted.

    Args:
        samples:
            np.array of integers or floats
        title:
            title of the plot
        transformation:
            transformation to do for displaying the min, median, max, and 66% and 95% low and high
            values for the HDI.
    """
    sim_min, sim_median, sim_max = np.quantile(samples, q=[0, 0.5, 1])
    sim_95_hdi_prob_low, sim_95_hdi_prob_hi = az.hdi(samples, hdi_prob=0.95)
    sim_66_hdi_prob_low, sim_66_hdi_prob_hi = az.hdi(samples, hdi_prob=0.66)

    def plot_text(x, label, above=True, factor=1):
        y = 0.018 + (0.025 * (factor - 1))
        if not above:
            y *= -1

        if transformation:
            x_formatted = format_number(transformation(x), places=decimals)
        else:
            x_formatted = format_number(x, places=decimals)

        return plt.text(
            x=x, y=y,
            s=f"{label}:\n{x_formatted}",
            ha='center', va='center', fontsize=9,
        )

    fig, ax = plt.subplots(1)
    plt.plot([sim_min, sim_max], [0, 0], color='gray')
    plt.plot([sim_95_hdi_prob_low, sim_95_hdi_prob_hi], [0, 0], color='black', linewidth=3)
    plt.plot([sim_66_hdi_prob_low, sim_66_hdi_prob_hi], [0, 0], color='black', linewidth=7)
    plt.plot(sim_median, 0, 'o', markersize=15, color='black')
    ax.set_yticklabels([])

    plot_text(x=sim_min, label='min')
    plot_text(x=sim_max, label="max")
    plot_text(x=sim_median, label="median", above=False)
    plot_text(x=sim_66_hdi_prob_low, label="HDI 66", factor=vertical_factor_66)
    plot_text(x=sim_66_hdi_prob_hi, label="HDI 66", factor=vertical_factor_66)
    plot_text(x=sim_95_hdi_prob_low, label="HDI 95", above=False, factor=vertical_factor_95)
    plot_text(x=sim_95_hdi_prob_hi, label="HDI 95", above=False, factor=vertical_factor_95)

    if ticks:
        plt.xticks(ticks)

    if transformation:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for index in range(len(labels)):
            labels[index] = format_number(
                transformation(float(labels[index].replace('−', '-'))),
                places=decimals
            )
        ax.set_xticks(ax.get_xticks())  # https://github.com/pandas-dev/pandas/issues/35684
        ax.set_xticklabels(labels)

    fig.set_size_inches(w=7, h=2.2)
    # plt.xlim((115, 195))
    if not title:
        title = "HDI"
    plt.suptitle(title)


def percentiles(values, p=[0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]) -> pd.DataFrame:
    return pd.DataFrame(dict(
        percentile=p,
        value=np.quantile(values, q=p)
    ))
