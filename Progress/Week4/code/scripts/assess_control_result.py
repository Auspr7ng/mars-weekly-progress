"""Versioned review of recorded results; never rewrites the original gate verdict."""
from pathlib import Path
import argparse,json,hashlib
WEEK=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--joystick-assessment',type=Path);args=p.parse_args()
    result=json.loads(args.result.read_text(encoding='utf-8'))
    gates=json.loads((WEEK/'config/control_acceptance_phase_aware.json').read_text(encoding='utf-8'))
    checkpoint=Path(result['checkpoint']);nominal=json.loads((checkpoint.parent/'dynamics/arm_home.json').read_text(encoding='utf-8'))
    weight_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def verify(data):
        assert hashlib.sha256(Path(data['checkpoint']).read_bytes()).hexdigest()==weight_hash,'Mismatched checkpoint in recorded evaluation'
        return data
    def read(key):return verify(json.loads((WEEK/result[key]).read_text(encoding='utf-8')))
    batch=read('evaluation');native=read('native_evaluation');joy=read('joystick_evaluation')
    old_joystick_verdict=joy['accepted']
    if args.joystick_assessment:joy=verify(json.loads(args.joystick_assessment.read_text(encoding='utf-8')))
    checks=dict(command_suite=batch['all_requested_programs_passed'] and batch['complete_control_suite'],
                native_feedback=native['all_requested_programs_passed'],joystick=joy['accepted'])
    rows=[]
    for key,fast in (('quality',False),('fast_quality',True)):
        report=read(key)
        for backend in ('gpu','native'):
            q=report[backend];height_limit=(nominal['root_height']*1000*gates['fast_height_fraction_of_nominal_root_height']) if fast else gates['slow_height_peak_to_peak_mm']
            values=dict(duration=q['duration_s']>=19.9,not_failed=not q.get('failed',False),no_termination=not q.get('termination_reasons',[]),no_self_contact=not q.get('self_contact_pairs',[]),
                roll=q['roll_peak_to_peak_deg']<=gates['roll_peak_to_peak_deg'],pitch=q['pitch_peak_to_peak_deg']<=gates['pitch_peak_to_peak_deg'],
                height=q['height_peak_to_peak_mm']<=height_limit,settled_support=q.get('settled_stance_speed_p95_m_s') is not None and q['settled_stance_speed_p95_m_s']<=gates['settled_stance_speed_p95_m_s'],
                touchdown=q.get('touchdown_speed_p95_m_s') is not None and q['touchdown_speed_p95_m_s']<=gates['touchdown_speed_p95_m_s'],
                impact=q['peak_foot_force_bodyweights']<=gates['peak_foot_force_bodyweights'],flight=q['flight_fraction']<=gates['flight_fraction'],
                support_balance=abs(q['left_right_support_fraction'][0]-q['left_right_support_fraction'][1])<=gates['max_support_fraction_difference'],
                swing_clearance=all(x is not None and x>=gates['minimum_swing_peak_mm'] for x in q['minimum_swing_peak_mm']))
            name=('fast_' if fast else 'slow_')+backend;checks[name]=all(values.values())
            rows.append(dict(name=name,checks=values,height_limit_mm=height_limit,original_all_contact_speed_p95_m_s=q.get('supported_sole_speed_p95_m_s'),
                settled_stance_speed_p95_m_s=q.get('settled_stance_speed_p95_m_s'),touchdown_speed_p95_m_s=q.get('touchdown_speed_p95_m_s'),height_peak_to_peak_mm=q['height_peak_to_peak_mm']))
    out=dict(original_result=str(args.result),original_engineering_verdict=result['control_engineering_checks_passed'],original_joystick_verdict=old_joystick_verdict,joystick_assessment=str(args.joystick_assessment) if args.joystick_assessment else None,criteria='config/control_acceptance_phase_aware.json',checks=checks,gait=rows,
             engineering_passed=all(checks.values()),visual_review_required=True,web_review_required=True,scope=gates['scope'])
    target=args.result.with_name(args.result.stem+'_assessment_v2.json');target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
