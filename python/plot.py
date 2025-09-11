import matplotlib

matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["text.usetex"] = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from python.error import theoretical_error, measured_error, inf_error

sns.set(font_scale=1.1)
sns.set_style(
    "whitegrid", {"axes.edgecolor": ".1", "grid.color": ".1", "grid.linestyle": "--"}
)


def expand_log_range(a, b, factor):
    c = np.power(a / b, factor)
    return c * b, a / c


def plot_rcmc(ax, data, type, reference, precision=15, max_states=6, interval=None):
    df_pop = pd.read_csv(
        f"result/{data}-pop-{type}-{precision}.txt", sep=" ", header=None
    )
    df_epoch = pd.read_csv(f"result/{data}-epoch-{precision}.csv")

    if interval:
        t_min, t_max = interval
        df_epoch = df_epoch.clip(t_min * 1e-5, t_max * 1e5)
    else:
        t_min, t_max = df_epoch.min().min(), df_epoch.max().max()

    ref_key = "time_eig" if reference == "eigen" else f"time_{reference}"

    plot_states = df_pop.max().sort_values(ascending=False).iloc[:max_states].index
    df_pop = df_pop[plot_states].set_index(df_epoch[ref_key])
    df_long = df_pop.reset_index().melt(ref_key)

    sns.scatterplot(
        ax=ax,
        data=df_long,
        x=ref_key,
        y="value",
        hue="variable",
        palette="deep",
        marker="o",
        s=20,
        edgecolor="none",
        legend=False,
    )


def plot_ode(ax, data, precision=200, max_states=6, interval=None):
    df = pd.read_csv(f"result/{data}-ode-{precision}.csv", index_col="time")
    df.rename(columns={c: int(c) for c in df.columns.values}, inplace=True)

    if interval:
        t_min, t_max = interval
        df = df.clip(t_min * 1e-5, t_max * 1e5)
    else:
        t_min, t_max = df.index.min(), df.index.max()

    plot_states = df.max().sort_values(ascending=False).iloc[:max_states].index
    df_long = df[plot_states].reset_index().melt("time")

    sns.lineplot(
        ax=ax,
        data=df_long,
        x="time",
        y="value",
        hue="variable",
        palette="deep",
        legend=False,
    )

    return (t_min, t_max)


def show_rcmc(
    data, type, reference, ax=None, precision=15, max_states=6, interval=None
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_xlabel(r"\rm $t$")
    ax.set_ylabel(r"\rm $q^{(k)}$")
    plot_rcmc(ax, data, type, reference, precision, max_states, interval)


def show_ode(data, ax=None, precision=200, max_states=6, interval=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    t_min, t_max = plot_ode(ax, data, precision, max_states, interval)
    ax.set_xlim(t_min, t_max)
    ax.set_xscale("log")
    ax.set_xlabel(r"\rm $t$")
    ax.set_ylabel(r"\rm $x(t)$")


def show_rcmc_ode(
    data,
    type,
    reference,
    ax=None,
    rcmc_precision=15,
    ode_precision=200,
    max_states=6,
    interval=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    t_min, t_max = plot_ode(ax, data, ode_precision, max_states, interval)
    plot_rcmc(ax, data, type, reference, rcmc_precision, max_states, (t_min, t_max))
    ax.set_xlim(t_min, t_max)
    ax.set_xscale("log")
    ax.set_xlabel(r"\rm $t$")
    ax.set_ylabel(r"\rm $x(t)$ and $q^{(k)}$")


def show_rcmc_ode_all_references(
    data, type, rcmc_precision=15, ode_precision=200, max_states=6, interval=None
):
    _, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 3))
    plt.subplots_adjust(wspace=0.1, hspace=0.32)
    for ref, ax in zip(["diag", "eigen", "gershgorin"], axes):
        show_rcmc_ode(
            data,
            type,
            ref,
            ax,
            rcmc_precision,
            ode_precision,
            max_states,
            interval,
        )
        ax.set_title(rf"\rm {ref}")


def plot_pi_error(ax, data, reference, rcmc_precision=15, ode_precision=200):
    df = pd.read_csv(f"result/{data}-epoch-15.csv")
    ref_key = "time_eig" if reference == "eigen" else f"time_{reference}"

    df["actual_A"] = measured_error(
        data, ode_precision, rcmc_precision, 15, ref_key, "A"
    )
    df["actual_B"] = measured_error(
        data, ode_precision, rcmc_precision, 15, ref_key, "B"
    )
    df["theory_A"] = theoretical_error(data, ode_precision, 15, ref_key, "A")
    df["theory_B"] = theoretical_error(data, ode_precision, 15, ref_key, "B")

    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="actual_A",
        label=r"\rm Actual A",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )
    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="actual_B",
        label=r"\rm Actual B",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )
    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="theory_A",
        label=r"\rm Theory A",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )
    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="theory_B",
        label=r"\rm Theory B",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )

    times = df[ref_key][(df[ref_key] != 0) & (df[ref_key] != np.inf)]
    x_range = (times.min(), times.max())

    errors = df[df[ref_key] != 0][["actual_A", "actual_B", "theory_A", "theory_B"]]
    errors = errors[errors > 0]
    y_range = (errors.min().min(), errors.max().max())

    return x_range, y_range


def show_pi_error_log(
    data,
    reference,
    ax=None,
    rcmc_precision=15,
    ode_precision=200,
    x_range=None,
    y_range=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_ret, y_ret = plot_pi_error(ax, data, reference, rcmc_precision, ode_precision)
    x_range = x_ret if x_range is None else x_range
    y_range = y_ret if y_range is None else y_range

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_range)
    ax.axhline(y=1.0, color="black", linestyle="-")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"\rm $\pi$-norm error")
    ax.set_title(rf"\rm {data} / {reference}")

    return x_range, y_range


def show_pi_error_semilog(
    data, reference, ax=None, rcmc_precision=15, ode_precision=200, x_range=None
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_ret, _ = plot_pi_error(ax, data, reference, rcmc_precision, ode_precision)
    x_range = x_ret if x_range is None else x_range

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"\rm $\pi$-norm error")
    ax.set_title(rf"\rm {data} / {reference}")

    return x_range


def show_pi_error_all_references(data, rcmc_precision=15, ode_precision=200):
    _, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 3))
    plt.subplots_adjust(wspace=0.1, hspace=0.32)
    x_min, x_max, y_min, y_max = 1.0, 1.0, 1.0, 1.0

    for ref, ax in zip(["diag", "eigen", "gershgorin"], axes):
        xr, yr = show_pi_error_log(data, ref, ax, rcmc_precision, ode_precision)
        x_min = min(x_min, xr[0])
        x_max = max(x_max, xr[1])
        y_min = min(y_min, yr[0])
        y_max = max(y_max, yr[1])

    for ax in axes:
        ax.set_xlim(expand_log_range(x_min, x_max, 1.05))
        ax.set_ylim(expand_log_range(y_min, y_max, 1.05))


def plot_inf_error(ax, data, reference, rcmc_precision=15, ode_precision=200):
    df = pd.read_csv(f"result/{data}-epoch-15.csv")
    ref_key = "time_eig" if reference == "eigen" else f"time_{reference}"

    df["A"] = inf_error(data, ode_precision, rcmc_precision, 15, ref_key, "A")
    df["B"] = inf_error(data, ode_precision, rcmc_precision, 15, ref_key, "B")

    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="A",
        label=r"\rm Actual A",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )
    sns.lineplot(
        ax=ax,
        data=df[df[ref_key] != 0],
        x=ref_key,
        y="B",
        label=r"\rm Actual B",
        marker="o",
        markersize=4,
        markeredgecolor="none",
    )

    times = df[ref_key][(df[ref_key] != 0) & (df[ref_key] != np.inf)]
    x_range = (times.min(), times.max())

    errors = df[df[ref_key] != 0][["A", "B"]]
    errors = errors[errors > 0]
    y_range = (errors.min().min(), errors.max().max())

    return x_range, y_range


def show_inf_error_semilog(
    data, reference, ax=None, rcmc_precision=15, ode_precision=200, x_range=None
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_ret, _ = plot_inf_error(ax, data, reference, rcmc_precision, ode_precision)
    x_range = x_ret if x_range is None else x_range

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"\rm $L_{\infty}$-norm error")
    ax.set_title(rf"\rm {data} / {reference}")

    return x_range


def stack_semilog_plots(data, reference, rcmc_precision=15, ode_precision=200):
    _, axes = plt.subplots(3, 1, sharex=True, figsize=(5, 9))
    plt.subplots_adjust(wspace=0.32, hspace=0.1)

    show_pi_error_semilog(data, reference, axes[0], rcmc_precision, ode_precision)
    show_inf_error_semilog(data, reference, axes[1], rcmc_precision, ode_precision)
    show_rcmc_ode(data, "B", reference, axes[2], rcmc_precision, ode_precision)

    axes[1].set_title("")
    axes[2].set_title("")
