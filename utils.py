import scipy.stats as st
import numpy as np

# function to display timer
def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:d}h{:02d}m{:02d}s".format(h,m,s)

def stats(scores):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    return np.mean(scores), float(low), float(up)
