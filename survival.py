import numpy as np
import pandas as pd

def kaplan_meier(duration: np.ndarray, event: np.ndarray):
    """Returns KM survival curve points (time, survival)."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)

    # sort by time
    order = np.argsort(duration)
    t = duration[order]
    e = event[order]

    uniq_times = np.unique(t)
    n = len(t)
    at_risk = n
    s = 1.0
    times = [0.0]
    surv = [1.0]

    for ut in uniq_times:
        mask = (t == ut)
        d = int(e[mask].sum())         # events at time
        c = int(mask.sum() - d)        # censored at time

        if at_risk > 0:
            if d > 0:
                s *= (1 - d / at_risk)
                times.append(float(ut))
                surv.append(float(s))
            at_risk -= (d + c)
        else:
            break
    return np.array(times), np.array(surv)

def concordance_index(duration, event, risk_score):
    """Harrell's C-index for right-censored data (basic implementation)."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)

    # comparable pairs: i experienced event, and j has larger time
    n = len(duration)
    conc = 0.0
    ties = 0.0
    total = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if duration[j] <= duration[i]:
                continue
            total += 1
            if risk[i] > risk[j]:
                conc += 1
            elif risk[i] == risk[j]:
                ties += 1
    if total == 0:
        return np.nan
    return (conc + 0.5 * ties) / total
