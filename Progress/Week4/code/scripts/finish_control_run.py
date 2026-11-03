"""Complete command-suite, native, gait and video checks after a bounded control run."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from datetime import datetime
WEEK=Path(__file__).resolve().parents[2]


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def quality_ok(q):
    speed=q.get('supported_sole_speed_p95_m_s')
    return (q['duration_s']>=19.9 and not q.get('failed',False) and not q.get('self_contact_pairs',[]) and
            q['roll_peak_to_peak_deg']<=8 and q['pitch_peak_to_peak_deg']<=6 and q['height_peak_to_peak_mm']<=10 and
            speed is not None and speed<=.035 and q['peak_foot_force_bodyweights']<=1.8 and q['flight_fraction']<=.01 and
            abs(q['left_right_support_fraction'][0]-q['left_right_support_fraction'][1])<=.08 and
            all(v is not None and v>=15 for v in q.get('minimum_swing_peak_mm',q['foot_height_max_mm'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--envs',type=int,default=100)
    p.add_argument('--profile',choices=['fast','responsive'],default='fast');args=p.parse_args()
    run=args.run_dir.resolve();state=WEEK/f'results/{args.tag}_postprocess.json'
    def mark(phase,**kw):write(state,dict(state=phase,pid=os.getpid(),run_dir=str(run),updated=datetime.now().isoformat(),**kw))
    def record_phase(phase,**kw):
        active=json.loads((WEEK/'config/active_run.json').read_text())
        if (WEEK/active['run_dir']).resolve()!=run:return
        path=WEEK/'config/execution_record.json';record=json.loads(path.read_text())
        training=json.loads((run/'status.json').read_text(encoding='utf-8'))
        record.update(training_running=training['state'] in ('initializing','running'),phase=phase,**kw);write(path,record)
    mark('waiting_for_training')
    try:
        if args.checkpoint:
            checkpoint=args.checkpoint.resolve()
            if checkpoint.parent!=run or not checkpoint.is_file():raise ValueError('Selected checkpoint must be inside this run')
        else:
            deadline=time.monotonic()+7200
            while time.monotonic()<deadline:
                try:training=json.loads((run/'status.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
                if training['state']=='completed':break
                if training['state'] in ('paused','failed'):
                    record_phase('control_training_'+training['state'])
                    mark('stopped',training_state=training['state']);return
                time.sleep(5)
            else:raise TimeoutError('Training exceeded the2h postprocessing wait')
            checkpoint=Path(training['checkpoint'])
        record_phase('control_training_completed_evaluating')
        spec=json.loads((run/'run.json').read_text());fast=spec['revision']>=22
        prefix='responsive' if args.profile=='responsive' else 'fast'
        limits=dict(forward_limit=.12,yaw_limit=.25,backward_limit=.06) if prefix=='responsive' else dict(forward_limit=.15,yaw_limit=.4,backward_limit=.06)
        if not fast and args.profile!='fast':raise ValueError('Responsive profile requires revision22 or later')
        def execute(stage,script,argv):
            mark(stage,checkpoint=str(checkpoint))
            with (run/f'control_{stage}.console.log').open('w',encoding='utf-8') as out,(run/f'control_{stage}.stderr.log').open('w',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-B','-u',str(WEEK/'code/scripts'/script),*argv],stdout=out,stderr=err,check=True,cwd=str(WEEK),timeout=1800)
        evaluation=WEEK/f'results/evaluations/{args.tag}_controls_100.json'
        programs=['idle','steady','speed_stop','turns','in_place','right_first','mixed_holdout']
        if fast:programs+=[prefix+'_steady',prefix+'_speed_stop',prefix+'_turns',prefix+'_in_place','responsive_backstep' if prefix=='responsive' else 'reverse_stop']
        execute('evaluating','evaluate_commands.py',['--checkpoint',str(checkpoint),'--envs',str(args.envs),'--seed','20261108','--programs',*programs,'--output',str(evaluation)])
        native=WEEK/f'results/evaluations/{args.tag}_native_controls.json'
        execute('native_feedback','evaluate_native_commands.py',['--checkpoint',str(checkpoint),'--programs',*programs,'--output',str(native)])
        execute('gait_quality','audit_gait_quality.py',['--checkpoint',str(checkpoint),'--speed','.06','--tag',args.tag+'_steady'])
        quality_path=WEEK/f'results/quality/{args.tag}_steady_quality.json';quality=json.loads(quality_path.read_text())
        fast_quality=None
        if fast:
            execute('fast_gait_quality','audit_gait_quality.py',['--checkpoint',str(checkpoint),'--speed',str(limits['forward_limit']),'--tag',args.tag+'_fast'])
            fast_quality=json.loads((WEEK/f'results/quality/{args.tag}_fast_quality.json').read_text())
        joystick=None
        if fast:
            joystick_path=WEEK/f'results/evaluations/{args.tag}_joystick_100.json'
            execute('joystick_feedback','evaluate_joystick_control.py',['--checkpoint',str(checkpoint),'--envs',str(args.envs),'--forward',str(limits['forward_limit']),'--yaw',str(limits['yaw_limit']),'--backward',str(limits['backward_limit']),'--output',str(joystick_path)])
            joystick=json.loads(joystick_path.read_text())
        execute('plotting','plot_control_review.py',['--evaluation',str(evaluation)])
        videos={}
        primary_program=prefix+'_speed_stop' if fast else 'speed_stop'
        for program in ((prefix+'_speed_stop',prefix+'_turns',prefix+'_in_place','responsive_backstep' if prefix=='responsive' else 'reverse_stop') if fast else ('speed_stop','turns','in_place')):
            video=WEEK/f'results/videos/{args.tag}_{program}.mp4'
            execute('recording_'+program,'play.py',['--checkpoint',str(checkpoint),'--program',program,'--dual-view','--record',str(video)])
            videos[program]=str(video.relative_to(WEEK))
        slow=WEEK/f'results/videos/{args.tag}_speed_stop_slow.mp4'
        execute('slow_motion','slow_motion.py',[str(WEEK/videos[primary_program]),str(slow)])
        preview=WEEK/f'results/videos/{args.tag}_stop_preview.gif'
        execute('preview','make_gif_preview.py',[str(WEEK/videos[primary_program]),str(preview),'--start','13','--end','19'])
        batch=json.loads(evaluation.read_text());cpu=json.loads(native.read_text())
        passed=bool(batch['all_requested_programs_passed'] and batch['complete_control_suite'] and cpu['all_requested_programs_passed'] and quality_ok(quality['gpu']) and quality_ok(quality['native']))
        if fast_quality is not None:passed=passed and quality_ok(fast_quality['gpu']) and quality_ok(fast_quality['native'])
        if joystick is not None:passed=passed and joystick['accepted']
        result=dict(state='completed',checkpoint=str(checkpoint),control_engineering_checks_passed=passed,control_capabilities_verified=False,visual_review_required=True,
                    evaluation=str(evaluation.relative_to(WEEK)),native_evaluation=str(native.relative_to(WEEK)),quality=str(quality_path.relative_to(WEEK)),
                    videos=videos,slow_video=str(slow.relative_to(WEEK)),preview=str(preview.relative_to(WEEK)),programs=batch['programs'])
        if fast:result.update(profile=args.profile,joystick_limits=limits,fast_quality=f'results/quality/{args.tag}_fast_quality.json',joystick_evaluation=f'results/evaluations/{args.tag}_joystick_100.json')
        write(WEEK/f'results/{args.tag}_result.json',result)
        rows=['# Joint control training results','',f'Checkpoint: `{checkpoint.parent.name}/{checkpoint.name}`.',
              '',f'Engineering checks: {passed}. Video review remains required; the accepted baseline is not replaced automatically.','',
              '| Program | Survived | Complete sequence passed |','| --- | --- | --- |']
        for c in batch['programs']:rows.append(f"| {c['program']} | {c['successes']}/{c['episodes']} | {c['capability_passed_episodes']}/{c['episodes']} |")
        rows+=['',f'[Full batch data](evaluations/{evaluation.name}) · [Independent native feedback](evaluations/{native.name}) · [Steady gait audit](quality/{quality_path.name})','']
        for program,video in videos.items():rows.append(f'[{program} policy recording](videos/{Path(video).name})')
        rows+=['',f'[Half-speed stopping recording](videos/{slow.name})','',f'![Stopping segment]({preview.as_posix()})','',
               'Heatmaps retain the Week3 distance response. The actor uses simulated base velocity; hardware needs a velocity estimator. This report does not establish hardware transfer.']
        report='\n'.join(rows)+'\n';(WEEK/f'results/{args.tag}_result.md').write_text(report,encoding='utf-8')
        active_path=WEEK/'config/active_run.json';active=json.loads(active_path.read_text())
        if (WEEK/active['run_dir']).resolve()==run:
            write(WEEK/'results/control_current_result.json',result);(WEEK/'results/control_current_result.md').write_text(report,encoding='utf-8')
            active.update(control_engineering_checks_passed=passed,control_capabilities_verified=False,control_evaluation=result['evaluation'],control_native_evaluation=result['native_evaluation'],
                          control_report='results/control_current_result.md',experimental_video=videos[primary_program],latest_intermediate_note=f'Control review complete: engineering_checks={passed}; see results/control_current_result.md')
            write(active_path,active)
            record_phase('control_round_review_complete',latest_evaluation=result['evaluation'])
        mark('completed',checkpoint=str(checkpoint),control_engineering_checks_passed=passed,report=f'results/{args.tag}_result.md')
    except BaseException as exc:
        record_phase('control_postprocessing_failed',postprocess_error=repr(exc))
        mark('failed',error=repr(exc));raise


if __name__=='__main__':main()
