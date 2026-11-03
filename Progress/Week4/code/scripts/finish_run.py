"""After a bounded training run, evaluate and record it locally without a new prompt."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from datetime import datetime

WEEK=Path(__file__).resolve().parents[2]


def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--speed',type=float,default=.08)
    p.add_argument('--checkpoint',type=Path,help='Evaluate this saved checkpoint directly, including a paused run')
    args=p.parse_args()
    run=args.run_dir.resolve()
    state_path=WEEK/f'results/{args.tag}_postprocess.json'
    evaluation=WEEK/f'results/evaluations/{args.tag}_final_100.json'
    video=WEEK/f'results/videos/{args.tag}_final.mp4'
    def status(phase,**kw):
        write_json(state_path,dict(state=phase,pid=os.getpid(),run_dir=str(run),updated=datetime.now().isoformat(),**kw))
    status('waiting_for_training')
    try:
        if args.checkpoint is not None:
            checkpoint=args.checkpoint.resolve()
            if checkpoint.parent!=run or not checkpoint.is_file():raise ValueError('Selected checkpoint must exist inside the requested run')
        else:
            deadline=time.monotonic()+7200
            while time.monotonic()<deadline:
                try:training=json.loads((run/'status.json').read_text(encoding='utf-8'))
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
                if training['state']=='completed':break
                if training['state'] in ('failed','paused'):
                    status('stopped',training_state=training['state']);return
                time.sleep(5)
            else:raise TimeoutError('Training did not finish within the 2-hour postprocess window')
            checkpoint=Path(training['checkpoint'])
        spec=json.loads((run/'run.json').read_text(encoding='utf-8'))
        def execute(name,argv):
            status(name,checkpoint=str(checkpoint))
            with (run/f'{name}.console.log').open('w',encoding='utf-8') as out,(run/f'{name}.stderr.log').open('w',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-B','-u',*argv],stdout=out,stderr=err,check=True,cwd=str(WEEK),timeout=1200)
        execute('evaluating',[str(WEEK/'code/scripts/evaluate.py'),'--checkpoint',str(checkpoint),'--envs','100','--seconds','20',
                              '--command',str(args.speed),'0','0','--seed','20261102','--output',str(evaluation)])
        report=json.loads(evaluation.read_text(encoding='utf-8'))
        quality=None;quality_passed=False
        if spec.get('revision',1)>=4:
            execute('auditing_quality',[str(WEEK/'code/scripts/audit_gait_quality.py'),'--checkpoint',str(checkpoint),'--speed',str(args.speed),'--tag',args.tag+'_final'])
            quality_path=WEEK/f'results/quality/{args.tag}_final_quality.json'
            quality=json.loads(quality_path.read_text(encoding='utf-8'))
            def screen(q):
                return (q['duration_s']>=19.9 and not q.get('failed',False) and not q.get('self_contact_pairs',[]) and
                        q['roll_peak_to_peak_deg']<=8 and q['pitch_peak_to_peak_deg']<=6 and q['height_peak_to_peak_mm']<=10 and
                        q['supported_sole_speed_p95_m_s']<=.035 and q['peak_foot_force_bodyweights']<=1.8 and q['flight_fraction']<=.01 and
                        abs(q['left_right_support_fraction'][0]-q['left_right_support_fraction'][1])<=.08 and
                        all(v is not None and v>=15 for v in q.get('minimum_swing_peak_mm',q['foot_height_max_mm'])))
            quality_passed=bool(screen(quality['gpu']) and screen(quality['native']))
        execute('recording',[str(WEEK/'code/scripts/play.py'),'--checkpoint',str(checkpoint),'--speed',str(args.speed),'--seconds','10','--dual-view','--record',str(video)])
        side=WEEK/f'results/videos/{args.tag}_side.mp4'
        execute('recording_side',[str(WEEK/'code/scripts/play.py'),'--checkpoint',str(checkpoint),'--speed',str(args.speed),'--seconds','10','--azimuth','90','--record',str(side)])
        slow=WEEK/f'results/videos/{args.tag}_side_slow.mp4'
        execute('exporting_slow_motion',[str(WEEK/'code/scripts/slow_motion.py'),str(side),str(slow)])
        numeric_passed=bool(report.get('walking_gate_passed',False))
        # Numerical screening is not visual acceptance; the user rejected the old gait.
        passed=False
        summary=dict(state='completed',checkpoint=str(checkpoint),walking_gate_passed=passed,numeric_gate_passed=numeric_passed,
                     quality_screen_passed=quality_passed,visual_review_required=True,
                     evaluation=str(evaluation),video=str(video),side_video=str(side),side_slow_video=str(slow),metrics=report,quality=quality)
        write_json(WEEK/'results/current_result.json',summary)
        text=(f'# Bounded training segment results\n\nCheckpoint: {checkpoint.parent.name}/{checkpoint.name}\n\n'
              f'- Test: {report["episodes"]} episodes x {report["seconds"]:.0f} seconds, command {args.speed:.2f} m/s\n'
              f'- Survived: {report["successes"]}/{report["episodes"]}\n'
              f'- Mean forward displacement: {report["mean_forward_displacement_m"]:.3f} m\n'
              f'- Velocity RMSE: {report["mean_velocity_rmse_m_s"]:.4f} m/s\n'
              f'- Mean maximum lateral drift: {report["mean_max_lateral_m"]:.3f} m\n'
              f'- Legacy numerical gate passed: {numeric_passed}\n'
              f'- Engineering quality screen: {quality_passed}\n'
              '- Visual gait acceptance: review required; no automatic pass\n\n'
              f'[Main view](videos/{video.name}) · [Side view](videos/{side.name}) · [Half-speed side view](videos/{slow.name}) · [Full evaluation](evaluations/{evaluation.name})\n\n'
              'These tests use flat ground and small randomized initial states. Review normal-speed and slow-motion footage for scuffing, sway and symmetry; survival and progress alone are insufficient.\n')
        if quality is not None:
            g,n=quality['gpu'],quality['native']
            def pair(values):return ' / '.join('No complete swing' if v is None else f'{v:.1f}' for v in values)
            text+=('\n## Single-start closed-loop quality audit\n\n'
                   '| Metric | GPU | Native MuJoCo | Engineering target |\n| --- | --- | --- | --- |\n'
                   f'| Roll peak-to-peak | {g["roll_peak_to_peak_deg"]:.1f}° | {n["roll_peak_to_peak_deg"]:.1f}° | ≤8° |\n'
                   f'| Pitch peak-to-peak | {g["pitch_peak_to_peak_deg"]:.1f}° | {n["pitch_peak_to_peak_deg"]:.1f}° | ≤6° |\n'
                   f'| Root-height peak-to-peak | {g["height_peak_to_peak_mm"]:.1f} mm | {n["height_peak_to_peak_mm"]:.1f} mm | ≤10 mm |\n'
                   f'| Minimum full-swing peak clearance (L/R) | {pair(g.get("minimum_swing_peak_mm",g["foot_height_max_mm"]))} mm | {pair(n.get("minimum_swing_peak_mm",n["foot_height_max_mm"]))} mm | Both ≥15 mm |\n'
                   f'| Supporting sole speed p95 | {g["supported_sole_speed_p95_m_s"]:.4f} m/s | {n["supported_sole_speed_p95_m_s"]:.4f} m/s | ≤0.035 m/s |\n'
                   f'| Peak foot force / body weight | {g["peak_foot_force_bodyweights"]:.2f} | {n["peak_foot_force_bodyweights"]:.2f} | ≤1.8 |\n'
                   '\nThis audit does not replace the 100-start batch test; heatmaps remain uncalibrated distance responses.\n')
        (WEEK/'results/current_result.md').write_text(text,encoding='utf-8')
        active_path=WEEK/'config/active_run.json'
        active=json.loads(active_path.read_text(encoding='utf-8'))
        if (WEEK/active['run_dir']).resolve()==run:
            active.update(walking_verified=False,quality_accepted=False,numeric_gate_passed=numeric_passed,quality_screen_passed=quality_passed,
                          evaluation=str(evaluation.relative_to(WEEK)),video=str(video.relative_to(WEEK)),side_video=str(side.relative_to(WEEK)))
            write_json(active_path,active)
            record_path=WEEK/'config/execution_record.json'
            record=json.loads(record_path.read_text(encoding='utf-8'))
            live=json.loads((run/'status.json').read_text(encoding='utf-8'))
            still_training=live['state'] in ('initializing','running')
            record.update(training_running=still_training,phase='intermediate_evaluation_complete' if still_training else 'bounded_round_complete',latest_evaluation=str(evaluation.relative_to(WEEK)),walking_gate_passed=passed)
            write_json(record_path,record)
        status('completed',checkpoint=str(checkpoint),walking_gate_passed=passed,evaluation=str(evaluation),video=str(video))
        print('POSTPROCESS_COMPLETED '+json.dumps(summary),flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
