"""Clock-time bins from unix seconds (UTC). Epoch day 0 (1970-01-01) was a THURSDAY -> dow = (day+3) % 7 with Monday=0."""
import numpy as np

N_BINS = {"daypart4_wkend": 8, "hour24": 24, "dow7": 7, "none": 1}


def time_bin(ts, scheme="daypart4_wkend"):
    ts = np.asarray(ts, dtype=np.int64)
    hour = (ts // 3600) % 24
    dow = ((ts // 86400) + 3) % 7
    if scheme == "daypart4_wkend":          # the 8 time states of the original idea: 4 dayparts x (weekday/weekend)
        return (hour // 6) + 4 * (dow >= 5)
    if scheme == "hour24":
        return hour
    if scheme == "dow7":
        return dow
    if scheme == "none":
        return np.zeros_like(ts)
    raise ValueError(scheme)
