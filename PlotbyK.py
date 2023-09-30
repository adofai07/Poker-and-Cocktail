from type import DoublePrecisionExponentialGeneratingFunctionOfStoppingTime, DoublePrecisionOrdinaryPowerSeriesGeneratingFunctionOfStoppingTime
import cplot
import tqdm
import numpy as np

def f(x):
    # print(type(x))
    # print(len(x))
    return np.log(x)

def egf(x):
    ret = list()

    for i in tqdm.trange(len(x)):
        ret.append(DoublePrecisionExponentialGeneratingFunctionOfStoppingTime(x[i], K=3, iters=50))
    
    return np.array(ret)

def opsgf(x):
    ret = list()

    for i in tqdm.trange(len(x)):
        ret.append(DoublePrecisionOrdinaryPowerSeriesGeneratingFunctionOfStoppingTime(x[i], K=3, iters=50))
        # ret.append(np.nan)
    
    return np.array(ret)

plt = cplot.plot(
    opsgf,
    (-5.0, +5.0, 400),
    (-5.0, +5.0, 400),
    # abs_scaling=lambda x: x / (x + 1),  # how to scale the lightness in domain coloring
    # contours_abs=2.0,
    # contours_arg=(-np.pi / 2, 0, np.pi / 2, np.pi),
    # emphasize_abs_contour_1: bool = True,
    # add_colorbars: bool = True,
    # add_axes_labels: bool = True,
    # saturation_adjustment: float = 1.28,
    # min_contour_length = None,
    # linewidth = None,
)
plt.show()