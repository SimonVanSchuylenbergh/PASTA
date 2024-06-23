import numpy as np
from General.MathUtils import mapRange
from scipy.stats.qmc import Sobol


def sobol_sample(
    n=1024,
    bounds=np.array([[7000, 30000], [-0.8, 0.8], [3.5, 5], [1, 300]]),
    round=True,
    seed=None,
) -> np.ndarray:
    """'
    Each sample is an array of length 4 containing [Teff, M, logg, vsini].
    """
    samples = Sobol(4, seed=seed).random_base2(int(np.rint(np.log2(n))))
    samples[:, 0] = np.clip(
        mapRange(samples[:, 0], 0, 1, bounds[0, 0], bounds[0, 1]),  # type: ignore
        bounds[0, 0],
        bounds[0, 1],
    )
    samples[:, 1] = np.clip(
        mapRange(samples[:, 1], 0, 1, bounds[1, 0], bounds[1, 1]),  # type: ignore
        bounds[1, 0],
        bounds[1, 1],
    )
    samples[:, 2] = np.clip(
        mapRange(samples[:, 2], 0, 1, bounds[2, 0], bounds[2, 1]),  # type: ignore
        bounds[2, 0],
        bounds[2, 1],
    )
    samples[:, 3] = np.clip(
        mapRange(samples[:, 3], 0, 1, bounds[3, 0], bounds[3, 1]),  # type: ignore
        bounds[3, 0],
        bounds[3, 1],
    )

    if round:
        samples[:, 0] = np.clip(np.rint(samples[:, 0]), bounds[0, 0], bounds[0, 1])
        samples[:, 1] = np.clip(np.round(samples[:, 1], 2), bounds[1, 0], bounds[1, 1])
        samples[:, 2] = np.clip(np.round(samples[:, 2], 2), bounds[2, 0], bounds[2, 1])
        samples[:, 3] = np.clip(np.rint(samples[:, 3]), bounds[3, 0], bounds[3, 1])
    return samples


def sobol_sample_no_vsini(
    n=1024,
    bounds=np.array([[7000, 30000], [-0.8, 0.8], [3.5, 5]]),
    vsini=0,
    round=True,
    seed=None,
) -> np.ndarray:
    """'
    Each sample is an array of length 4 containing [Teff, M, logg, vsini].
    """
    samples = np.zeros((n, 4))
    samples[:, :3] = Sobol(3, seed=seed).random_base2(int(np.rint(np.log2(n))))
    samples[:, 0] = np.clip(
        mapRange(samples[:, 0], 0, 1, bounds[0, 0], bounds[0, 1]),  # type: ignore
        bounds[0, 0],
        bounds[0, 1],
    )
    samples[:, 1] = np.clip(
        mapRange(samples[:, 1], 0, 1, bounds[1, 0], bounds[1, 1]),  # type: ignore
        bounds[1, 0],
        bounds[1, 1],
    )
    samples[:, 2] = np.clip(
        mapRange(samples[:, 2], 0, 1, bounds[2, 0], bounds[2, 1]),  # type: ignore
        bounds[2, 0],
        bounds[2, 1],
    )

    if round:
        samples[:, 0] = np.clip(np.rint(samples[:, 0]), bounds[0, 0], bounds[0, 1])
        samples[:, 1] = np.clip(np.round(samples[:, 1], 2), bounds[1, 0], bounds[1, 1])
        samples[:, 2] = np.clip(np.round(samples[:, 2], 2), bounds[2, 0], bounds[2, 1])
        samples[:, 3] = np.rint(vsini)
    else:
        samples[:, 3] = vsini

    return samples
