"""Display the active run without importing the training framework."""
from pathlib import Path
import json

week=Path(__file__).resolve().parents[2]
active=json.loads((week/'config/active_run.json').read_text(encoding='utf-8'))
run=week/active['run_dir']
status=json.loads((week/active['status_path']).read_text(encoding='utf-8'))
manifest_path=week/'config/accepted_policy.json'
if manifest_path.exists():
    accepted=json.loads(manifest_path.read_text(encoding='utf-8'))
    if accepted.get('accepted') and accepted.get('web_interaction_verified') and active.get('control_capabilities_verified'):
        print('FINAL ACCEPTANCE: COMPLETE (flat-ground simulation)')
        print('OPEN DEMO: http://127.0.0.1:8086/ or start_demo.bat')
        print('ACCEPTED CHECKPOINT:',accepted['checkpoint'])
        if status['state']=='paused':print('Training stopped after a verified saved checkpoint was selected; the accepted demo is ready.')
print('ACTIVE RUN:',active['run_dir'])
print('STATE:',status['state'])
print('TASK REVISION:',active['task_revision'])
if 'iteration' in status:print('CHECKPOINT ITERATION:',status['iteration'])
spec=json.loads((run/'run.json').read_text(encoding='utf-8'))
if 'iteration' in status and not spec.get('resume'):print('ITERATIONS COMPLETED:',status['iteration']+1,'/',spec['iterations'])
if 'environment_transitions' in status:print('ENVIRONMENT TRANSITIONS:',status['environment_transitions'])
if 'checkpoint' in status:print('CHECKPOINT:',status['checkpoint'])
print('FULL WALKING TASK VERIFIED:',active['walking_verified'])
if 'fixed_speed_baseline_verified' in active:print('FIXED-SPEED BASELINE VERIFIED:',active['fixed_speed_baseline_verified'])
if 'numeric_gate_passed' in active:print('NUMERIC GATE:',active['numeric_gate_passed'])
if 'quality_screen_passed' in active:print('GAIT QUALITY SCREEN:',active['quality_screen_passed'])
if 'control_capabilities_verified' in active:print('SPEED/STOP/TURN VERIFIED:',active['control_capabilities_verified'])
if 'standing_and_stop_numeric_checks_passed' in active:print('STANDING/STOP NUMERIC CHECKS (SIMULATION):',active['standing_and_stop_numeric_checks_passed'])
if 'control_report' in active:print('LATEST CONTROL REPORT:',active['control_report'])
if active.get('selected_checkpoint'):print('SELECTED REPLAY CHECKPOINT:',active['selected_checkpoint'])
if 'verification_scope' in active:print('VERIFICATION SCOPE:',active['verification_scope'])
if 'latest_intermediate_quality' in active:print('LATEST QUALITY AUDIT:',active['latest_intermediate_quality'])
if 'latest_intermediate_note' in active:print('QUALITY NOTE:',active['latest_intermediate_note'])
if 'postprocess_status' in active and (week/active['postprocess_status']).exists():
    post=json.loads((week/active['postprocess_status']).read_text(encoding='utf-8'))
    print('RESULT PROCESSING:',post['state'])
    if 'checkpoint' in post:print('RESULT CHECKPOINT:',post['checkpoint'])
    if 'error' in post:print('RESULT ERROR:',post['error'])
if (run/'failure.json').exists():print('ERROR:',(run/'failure.json').read_text(encoding='utf-8'))
print('\nRECENT LOG:')
if (run/'console.log').exists():
    print('\n'.join((run/'console.log').read_text(encoding='utf-8',errors='replace').splitlines()[-20:]))
