"""Plot measured test traces without changing the model or test results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]/'results'
fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
for name,label in [('push_forward_5N','5 N / aligned'),('push_forward_10N','10 N / aligned'),
                   ('push_forward_20N','20 N / aligned'),('casters_90deg_push10N','10 N / 90-degree casters')]:
    data=json.loads((OUT/(name+'.json')).read_text())['trace']
    axes[0].plot([r['t'] for r in data],[r['linear_velocity'][0] for r in data],label=label)
data=json.loads((OUT/'casters_90deg_push10N.json').read_text())['trace']
for i,label in enumerate(('front-left','front-right','rear-left','rear-right')):
    axes[1].plot([r['t'] for r in data],[np.degrees(r['swivel_rad'][i]) for r in data],label=label)
for ax in axes:
    ax.axvspan(2,4,color='orange',alpha=.13,label='force on' if ax is axes[0] else None)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9,loc='upper right')
axes[0].set_ylabel('Forward speed (m/s)')
axes[1].set_ylabel('Caster angle (degrees)')
axes[1].set_xlabel('Simulation time (s)')
axes[0].set_title('Cart diagnostic copy: speed under handle force')
axes[1].set_title('Initially transverse casters: limited reorientation under 10 N')
fig.suptitle('Assumed fork mass: 0.05 kg each. Original XML fails to compile.',fontsize=12)
fig.savefig(OUT/'motion_comparison.png',dpi=160)
