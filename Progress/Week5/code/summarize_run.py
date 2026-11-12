"""Summarize observed input/physics evidence; does not generate synthetic input."""
import json
from pathlib import Path
import time

WEEK = Path(__file__).resolve().parents[1]


def main():
    trace = [json.loads(line) for line in (WEEK/'results/simulation_trace.jsonl').read_text().splitlines()]
    # Each server starts data.time at zero; resets also delimit episodes.
    cut = max([i for i in range(1, len(trace)) if trace[i]['simulation_time_s'] < trace[i-1]['simulation_time_s']] or [0])
    rows = trace[cut:]
    inputs = [json.loads(line) for line in (WEEK/'results/input_trace.jsonl').read_text().splitlines()]
    inputs = [r for r in inputs if r['time'] >= rows[0]['time']]
    def count(axis, sign):
        return sum(r['command'][axis]*sign > 0 for r in rows)
    zero = [r for r in rows[-20:] if not any(r['command'])]
    server = json.loads((WEEK/'results/server_info.json').read_text(encoding='utf-8'))
    result = dict(created_at_unix=time.time(), input_backend='pygame_sdl_standard',
                  walking_input=server.get('walking_input'), command_mapper=server.get('command_mapper'),
                  right_stick=server.get('right_stick'),
                  physical_controller_detected=any(r['input']['connected'] for r in inputs),
                  device=rows[-1]['input']['device'], samples=len(rows),
                  forward_samples=count(0, 1), reverse_samples=count(0, -1),
                  left_turn_samples=count(2, 1), right_turn_samples=count(2, -1),
                  diagonal_walk_turn_samples=sum(bool(r['command'][0] and r['command'][2]) for r in rows),
                  b_button_observed=any(r['input']['b'] for r in inputs),
                  menu_button_observed=any(r['input']['menu'] for r in inputs),
                  root_x_range_m=max(r['root_xyz_m'][0] for r in rows)-min(r['root_xyz_m'][0] for r in rows),
                  root_y_range_m=max(r['root_xyz_m'][1] for r in rows)-min(r['root_xyz_m'][1] for r in rows),
                  actual_forward_velocity_range_m_s=[min(r['actual_forward_m_s'] for r in rows), max(r['actual_forward_m_s'] for r in rows)],
                  actual_yaw_velocity_range_rad_s=[min(r['actual_yaw_rad_s'] for r in rows), max(r['actual_yaw_rad_s'] for r in rows)],
                  all_commands_bounded=all(-.060001 <= r['command'][0] <= .120001 and r['command'][1] == 0. and abs(r['command'][2]) <= .250001 for r in rows),
                  zero_command_samples_at_end=len(zero),
                  mean_abs_forward_at_zero_m_s=sum(abs(r['actual_forward_m_s']) for r in zero)/len(zero) if zero else None,
                  mean_abs_yaw_at_zero_rad_s=sum(abs(r['actual_yaw_rad_s']) for r in zero)/len(zero) if zero else None,
                  termination_observed=any('| END:' in r['status'] for r in rows),
                  physical_disconnect_reconnect_tested=False,
                  evidence=['results/input_trace.jsonl','results/simulation_trace.jsonl','progress/media/xbox_left_stick.jpg'],
                  note='Observed controller inputs and native MuJoCo responses; a short interactive check, not a new full policy acceptance.')
    (WEEK/'results/run_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
