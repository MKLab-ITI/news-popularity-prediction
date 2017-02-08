__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def ci_lower_bound(pos, n, confidence):
    if n == 0:
        return 0.0
    # z = pnormaldist(1 - (1 - confidence)/2)
    z = 1.96
    phat = 1.0*pos/n
    lower_bound = (phat + (z*z)/(2*n) - z * np.sqrt((phat * (1 - phat) + (z*z)/(4*n))/n))/(1 + (z*z)/n)
    return lower_bound
