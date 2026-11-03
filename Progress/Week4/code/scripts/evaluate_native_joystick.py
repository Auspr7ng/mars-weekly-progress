"""Replay the shared joystick lifecycle through native physics and CPU inference."""
import bootstrap
bootstrap.configure_warp()
import argparse,json,sys
from pathlib import Path
from evaluate_joystick_control import STAGES
from evaluate_native_commands import main as native_main
from sunday_locomotion.arrow_control import ArrowState,joystick_to_command
from sunday_locomotion.command_programs import PROGRAMS


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forward',type=float,default=.12);p.add_argument('--yaw',type=float,default=.25);p.add_argument('--backward',type=float,default=.06);args=p.parse_args()
    limits=dict(forward_limit=args.forward,yaw_limit=args.yaw,backward_limit=args.backward);state=ArrowState();stages=[]
    for end,vector in STAGES:
        if vector is None:state.release('end')
        else:state.drag('start',vector,1)
        command,_=joystick_to_command(state.vector,state.held,**limits)
        stages.append((end,float(command[0]),float(command[2])))
    PROGRAMS['web_joystick']=stages
    sys.argv=[sys.argv[0],'--checkpoint',str(args.checkpoint),'--output',str(args.output),'--actor-device','cpu','--programs','web_joystick']
    native_main()
    result=json.loads(args.output.read_text(encoding='utf-8'));result.update(joystick_limits=limits,scope='Native CPU physics and CPU policy follow the exact shared60-second hold/release command sequence. Detailed one-second burst thresholds are additionally tested in the GPU100-start joystick audit.')
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
