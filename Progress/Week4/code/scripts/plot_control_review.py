"""Plot measured command following and foot motion, including stop intervals."""
import bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sunday_locomotion.command_programs import segments

p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);args=p.parse_args()
report=json.loads(args.evaluation.read_text(encoding='utf-8'))
base=args.evaluation.with_suffix('');speed=np.load(str(base)+'_speed_stop_trace.npz');turn=np.load(str(base)+'_turns_trace.npz')
fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
def measured(ax,time,data,label,color):
    ax.plot(time,data.mean(axis=1),label=label,color=color,linewidth=1.2)
    ax.fill_between(time,np.percentile(data,5,axis=1),np.percentile(data,95,axis=1),color=color,alpha=.13)
def stops(ax,program):
    for segment in segments(program):
        if segment['kind']=='stop':ax.axvspan(segment['start'],segment['end'],color='#94a3b8',alpha=.2)
t=speed['time'];ax=axes[0]
ax.plot(t,speed['command'][:,:,0].mean(axis=1),'--',color='#172032',label='Filtered forward command')
measured(ax,t,speed['velocity'][:,:,0],'Measured forward velocity','#2563eb');stops(ax,'speed_stop')
ax.set(ylabel='Forward velocity (m/s)',title='Speed / stop / restart: shaded regions request zero motion');ax.legend(loc='upper left',ncol=2)
ax=axes[1]
measured(ax,t,speed['sole_low'][:,:,0]*1000,'Left sole clearance','#2563eb')
measured(ax,t,speed['sole_low'][:,:,1]*1000,'Right sole clearance','#dc2626');stops(ax,'speed_stop')
ax.axhline(0,color='black',linewidth=.7);ax.set(ylabel='Lowest sole height (mm)',title='Measured foot lifting: shaded regions request zero motion');ax.legend(loc='upper left',ncol=2)
t=turn['time'];ax=axes[2]
ax.plot(t,turn['command'][:,:,2].mean(axis=1),'--',color='#172032',label='Filtered turn command')
measured(ax,t,turn['omega'][:,:,2],'Measured yaw rate','#7c3aed');stops(ax,'turns')
ax.set(ylabel='Yaw rate (rad/s)',xlabel='Physics time (s)',title='Walking turns and in-place yaw');ax.legend(loc='upper left',ncol=2)
for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,30)
checkpoint=Path(report['checkpoint']).stem
count=next(x['episodes'] for x in report['programs'] if x['program']=='speed_stop')
fig.suptitle(f"R{report['revision']} {checkpoint}: actual policy trajectories, {count} randomized initial states\nLines: ensemble means; bands: 5th-95th percentiles",fontsize=12)
out=args.evaluation.with_suffix('.png');fig.savefig(out,dpi=160);plt.close(fig);print(out)
