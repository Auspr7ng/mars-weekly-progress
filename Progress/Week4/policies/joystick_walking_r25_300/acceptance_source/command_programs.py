"""Repeatable command sequences shared by tests and actual policy recordings."""
PROGRAMS={
    'responsive_backstep':[(3,0,0),(9,-.04,0),(13,0,0),(19,-.06,0),(24,0,0)],
    'responsive_steady':[(20,.12,0)],
    'responsive_speed_stop':[(4,.06,0),(9,.09,0),(14,.12,0),(20,0,0),(26,.10,0),(30,0,0)],
    'responsive_turns':[(4,.08,0),(10,.08,.25),(13,0,0),(19,.08,-.25),(22,0,0),(27,0,.25),(31,0,0)],
    'responsive_in_place':[(3,0,0),(9,0,.25),(12,0,0),(18,0,-.25),(24,0,0)],
    'responsive_reverse':[(3,0,0),(9,-.02,0),(13,0,0),(19,-.03,0),(24,0,0)],
    'fast_steady':[(20,.15,0)],
    'fast_speed_stop':[(4,.06,0),(9,.12,0),(14,.15,0),(20,0,0),(26,.10,0),(30,0,0)],
    'fast_turns':[(4,.10,0),(10,.10,.40),(13,0,0),(19,.10,-.40),(22,0,0),(27,0,.40),(31,0,0)],
    'fast_in_place':[(3,0,0),(9,0,.40),(12,0,0),(18,0,-.40),(24,0,0)],
    'reverse_stop':[(3,0,0),(9,-.05,0),(13,0,0),(19,-.06,0),(24,0,0)],
    'idle':[(10,0,0)],
    'steady':[(20,.06,0)],
    'speed_stop':[(4,.06,0),(9,.03,0),(14,.08,0),(21,0,0),(27,.06,0),(30,0,0)],
    'turns':[(4,.06,0),(10,.05,.12),(13,0,0),(19,.05,-.12),(22,0,0),(26,0,.12),(30,0,0)],
    'in_place':[(3,0,0),(9,0,.12),(12,0,0),(18,0,-.12),(24,0,0)],
    'smoke':[(3,.06,0),(6,0,0),(10,.05,.08),(12,.05,-.08),(14,0,0)],
    'right_first':[(3,0,0),(9,0,-.12),(12,0,0),(18,0,.12),(24,0,0)],
    'mixed_holdout':[(4,.04,0),(9,.07,.10),(13,0,0),(18,0,-.10),(22,.05,0),(27,.03,-.10),(32,0,0)],
}


def segments(name):
    start=0.;result=[]
    for end,speed,yaw in PROGRAMS[name]:
        kind='stop' if speed==0 and yaw==0 else ('in_place' if speed==0 else ('turn' if yaw else 'walk'))
        result.append(dict(start=start,end=float(end),command=[speed,0.,yaw],kind=kind));start=float(end)
    return result


def command_at(name,time):
    for segment in segments(name):
        if time<segment['end']-1e-9:return segment['command']
    return list(segments(name)[-1]['command'])


def duration(name):return float(PROGRAMS[name][-1][0])
