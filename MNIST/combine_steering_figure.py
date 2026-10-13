import matplotlib
matplotlib.use('Agg')

import runpy
import numpy as np
import matplotlib.pyplot as plt

STEERING_EFFECT_KAPPAS = [100, 0.003, 0.002, 0.001, 0.0001]
MAIN_KAPPA = 0.000001

print("Running main.py as-is ...")
main_ns = runpy.run_path('main.py', run_name='__main__')
IMG_SIZE = main_ns['IMG_SIZE']
main_best_image = ((main_ns['samples'][main_ns['best_idx']].reshape(IMG_SIZE, IMG_SIZE) + 1.0) / 2.0).clip(0, 1)
plt.close('all')

print("Running steering_effect.py as-is ...")
steer_ns = runpy.run_path('steering_effect.py', run_name='__main__')
steer_axes = steer_ns['axes']
steer_kappas = steer_ns['KAPPAS']

images = {}
for ax, kappa in zip(steer_axes, steer_kappas):
    if kappa in STEERING_EFFECT_KAPPAS:
        images[kappa] = ax.images[0].get_array()
images[MAIN_KAPPA] = main_best_image
plt.close('all')

ordered_kappas = STEERING_EFFECT_KAPPAS + [MAIN_KAPPA]

fig, axes = plt.subplots(1, len(ordered_kappas), figsize=(4 * len(ordered_kappas), 4))
for ax, kappa in zip(axes, ordered_kappas):
    ax.imshow(images[kappa], cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(rf'$\kappa = {kappa:g}$', fontsize=26)

fig.tight_layout()
fig.savefig('Figures/mnist_steering_effect_combined.pdf', bbox_inches='tight')
print("Saved -> Figures/mnist_steering_effect_combined.pdf")
