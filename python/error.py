import gmpy2
import math
import numpy as np
import pandas as pd

from python.load import load_rcm, load_eigen


def get_closest_index(df, value):
    if value == np.inf:
        return df.index.max()
    else:
        return (df.index.to_series() - value).abs().idxmin()


def pi_norm(x, pi):
    s = 0.0
    for a, b in zip(x, pi):
        s += a**2 / b
    return np.sqrt(float(s))


def measured_error(data, eig_precision, rcmc_precision, epoch_precision, time, type):
    gmpy2.get_context().precision = eig_precision

    _, pi = load_rcm(f"data/{data}.txt", gmpy2.mpfr)
    p_norm = 1.0 / math.sqrt(pi[0])

    df_rcmc = pd.read_csv(
        f"result/{data}-pop-{type}-{rcmc_precision}.txt",
        sep=" ",
        header=None,
        dtype=str,
    )
    df_ode = pd.read_csv(
        f"result/{data}-ode-epoch-{eig_precision}-15.csv", index_col="time", dtype=str
    )
    df_epoch = pd.read_csv(f"result/{data}-epoch-{epoch_precision}.csv")

    for col in df_rcmc.columns:
        df_rcmc[col] = df_rcmc[col].apply(gmpy2.mpfr)

    for col in df_ode.columns:
        df_ode[col] = df_ode[col].apply(gmpy2.mpfr)

    errors = []
    for i, row in df_rcmc.iterrows():
        t = df_epoch.loc[i, time]
        Vp = row.to_numpy()
        Ep = df_ode.loc[get_closest_index(df_ode, t)].to_numpy()
        if Ep.ndim > 1:
            Ep = Ep[0, :]
        errors.append(pi_norm(Vp - Ep, pi) / p_norm)

    return errors


# Specially cares the case where t = inf and lam = 0.0
def exp(t, lam):
    return math.exp(t * lam) if lam < 0.0 else 1.0


def alpha(rho_D, lam, t):
    fraction = rho_D / abs(lam) if lam < 0.0 else np.inf
    return fraction + exp(t, lam)


def beta(sigma_KSS, lam, t, type, K_norm=None):
    B_error = abs(lam) / sigma_KSS + 1.0 - exp(t, lam)
    if type == "A":
        return B_error + math.sqrt(2.0 * K_norm * abs(lam)) / sigma_KSS
    elif type == "B":
        return B_error
    else:
        raise ValueError(f"Unknown type: {type}")


def off_norm(K, pi):
    norm = 0.0
    for i, j, v in K:
        if i != j:
            norm = max(norm, abs(v) / pi[i])
    return norm


def theoretical_error(data, eig_precision, epoch_precision, time, type):
    gmpy2.get_context().precision = eig_precision

    K, pi = load_rcm(f"data/{data}.txt")
    df_epoch = pd.read_csv(f"result/{data}-epoch-{epoch_precision}.csv")
    lam, U = load_eigen(f"result/{data}-eig-{eig_precision}.txt")

    p_norm = 1.0 / math.sqrt(pi[0])
    c = U[0, :] / pi[0]
    K_norm = off_norm(K, pi)

    errors = []
    for _, row in df_epoch.iterrows():
        t = row[time]
        rho_D = row["rho_D"]
        sigma_KSS = row["sigma_KSS"]

        error = 0.0
        for k, l in enumerate(lam):
            a = alpha(rho_D, l, t)
            b = beta(sigma_KSS, l, t, type, K_norm)
            error += abs(c[k]) * min(1.0, a, b)

        errors.append(min(error / p_norm, 1))

    return errors


def expected_error(data, eig_precision, epoch_precision, time, type):
    gmpy2.get_context().precision = eig_precision

    K, pi = load_rcm(f"data/{data}.txt")
    df_epoch = pd.read_csv(f"result/{data}-epoch-{epoch_precision}.csv")
    lam, _ = load_eigen(f"result/{data}-eig-{eig_precision}.txt")

    n = len(pi)
    K_norm = off_norm(K, pi)

    errors = []
    for _, row in df_epoch.iterrows():
        t = row[time]
        rho_D = row["rho_D"]
        sigma_KSS = row["sigma_KSS"]

        error = 0.0
        for l in lam:
            a = alpha(rho_D, l, t)
            b = beta(sigma_KSS, l, t, type, K_norm)
            error += min(1.0, a, b) ** 2

        errors.append(min(math.sqrt(error / n), 1))

    return errors


def inf_norm(x):
    return float(max(x))


def inf_error(data, eig_precision, rcmc_precision, epoch_precision, time, type):
    gmpy2.get_context().precision = eig_precision

    df_rcmc = pd.read_csv(
        f"result/{data}-pop-{type}-{rcmc_precision}.txt",
        sep=" ",
        header=None,
        dtype=str,
    )
    df_ode = pd.read_csv(
        f"result/{data}-ode-epoch-{eig_precision}-15.csv", index_col="time", dtype=str
    )
    df_epoch = pd.read_csv(f"result/{data}-epoch-{epoch_precision}.csv")

    for col in df_rcmc.columns:
        df_rcmc[col] = df_rcmc[col].apply(gmpy2.mpfr)

    for col in df_ode.columns:
        df_ode[col] = df_ode[col].apply(gmpy2.mpfr)

    errors = []
    for i, row in df_rcmc.iterrows():
        t = df_epoch.loc[i, time]
        Vp = row.to_numpy()
        Ep = df_ode.loc[get_closest_index(df_ode, t)].to_numpy()
        if Ep.ndim > 1:
            Ep = Ep[0, :]
        errors.append(inf_norm(Vp - Ep))

    return errors
