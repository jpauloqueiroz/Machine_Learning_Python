from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs, ys):
    numerator = (np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)
    denominator = np.mean(xs)**2 - np.mean(xs**2)
    return numerator / denominator

m = best_fit_slope(xs, ys)
print(m)