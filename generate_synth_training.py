import numpy as np
from tqdm.auto import tqdm

from definitions import HERMESNET, PROJECT_ROOT
from GSSP.SobolSampler import sobol_sample
from interpolator import get_cached_interpolator, get_in_mem_interpolator
from nmf.load_labeled_data import load_observed_npy


output = HERMESNET / "synth_interpolated_sobol_log_train"
# output = HERMESNET / "synth_interpolated_mimick2"
output.mkdir(exist_ok=True)
print(output)


def label_to_filename(label):
    return f"{round(label[0])}_{round(label[1]*1e3)}_{round(label[2]*1e3)}_{round(label[3])}.npy"


interpolator = get_cached_interpolator("/scratch/ragnarv/fine_grid")

labels = sobol_sample(
    n=1024 * 16,
    # n=64,
    # bounds=np.array([[np.log10(6000), np.log10(30000)], [-0.7, 0.7], [2.5, 4.5], [1, 350]]),
    bounds=np.array([[np.log10(6000), np.log10(30000)], [-0.8, 0.8], [2.5, 5.0], [1, 370]]),
    round=False,
)
labels = np.c_[labels, np.zeros(labels.shape[0])]
labels[:, 0] = np.round(10**(labels[:, 0]), 0)
labels[:, 1] = np.round(labels[:, 1], 3)
labels[:, 2] = np.round(labels[:, 2], 3)
labels[:, 3] = np.round(labels[:, 3], 0)
labels[:, 4] = 0


# interpolator = get_cached_interpolator()
bounded_labels = labels[[interpolator.is_within_bounds(*x) for x in tqdm(labels)]][:8192]
print(len(bounded_labels))
wl = np.load("wl.npy")
np.save(output / "wl.npy", wl)
for i in tqdm(list(range(len(bounded_labels) // 95))):
    selection = bounded_labels[i * 95 : (i + 1) * 95]
    models = np.array(
        interpolator.produce_model_bulk(
            wl, [tuple(x) for x in selection], progress=False
        )
    )
    for l, m in zip(selection, models):
        np.save(output / label_to_filename(l), m)
