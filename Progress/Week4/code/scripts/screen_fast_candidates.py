"""Screen saved fast-control candidates while the bounded training continues."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime

WEEK=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--iterations',nargs='+',type=int,required=True);p.add_argument('--tag-prefix',default='control_r22');args=p.parse_args()
    run=args.run_dir.resolve();progress=WEEK/'results/fast_candidate_screen_status.json'
    def mark(state,**kw):
        progress.write_text(json.dumps(dict(state=state,run_dir=str(run),tag_prefix=args.tag_prefix,updated=datetime.now().isoformat(),**kw),indent=2),encoding='utf-8')
    try:
        for iteration in args.iterations:
            checkpoint=run/f'model_{iteration}.pt';deadline=time.monotonic()+7200
            mark('waiting_for_checkpoint',iteration=iteration)
            while time.monotonic()<deadline:
                try:status=json.loads((run/'status.json').read_text(encoding='utf-8'))
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
                if checkpoint.is_file() and status.get('iteration',-1)>=iteration:break
                if status['state'] in ('paused','failed','completed'):
                    mark('stopped',reason='Training ended before the requested checkpoint',iteration=iteration);return
                time.sleep(5)
            else:raise TimeoutError('Candidate checkpoint wait exceeded2h')
            for profile in ('fast','responsive'):
                tag=f'{args.tag_prefix}_candidate{iteration}'+('_responsive' if profile=='responsive' else '')
                programs=[profile+'_steady',profile+'_speed_stop',profile+'_turns',profile+'_in_place',
                          'responsive_backstep' if profile=='responsive' else 'reverse_stop','in_place']
                output=WEEK/f'results/evaluations/{tag}_native_screen.json'
                mark('native_screen',iteration=iteration,profile=profile,checkpoint=str(checkpoint))
                with (run/f'{tag}.screen.console.log').open('w',encoding='utf-8') as out,(run/f'{tag}.screen.stderr.log').open('w',encoding='utf-8') as err:
                    subprocess.run([sys.executable,'-B','-u',str(WEEK/'code/scripts/evaluate_native_commands.py'),
                        '--checkpoint',str(checkpoint),'--programs',*programs,'--output',str(output)],stdout=out,stderr=err,check=True,timeout=1800)
                screen=json.loads(output.read_text(encoding='utf-8'))
                if not screen['all_requested_programs_passed']:
                    mark('candidate_rejected',iteration=iteration,profile=profile,screen=str(output));continue
                mark('full_candidate_review',iteration=iteration,profile=profile)
                with (run/f'{tag}.review.console.log').open('w',encoding='utf-8') as out,(run/f'{tag}.review.stderr.log').open('w',encoding='utf-8') as err:
                    subprocess.run([sys.executable,'-B','-u',str(WEEK/'code/scripts/finish_control_run.py'),
                        '--run-dir',str(run),'--tag',tag,'--checkpoint',str(checkpoint),'--profile',profile,'--envs','100'],stdout=out,stderr=err,check=True,timeout=3600)
                result=json.loads((WEEK/f'results/{tag}_result.json').read_text(encoding='utf-8'))
                if result['control_engineering_checks_passed']:
                    mark('candidate_ready_for_visual_review',iteration=iteration,profile=profile,result=f'results/{tag}_result.json');return
        mark('screens_completed_no_accepted_candidate')
    except BaseException as exc:mark('failed',error=repr(exc));raise


if __name__=='__main__':main()
