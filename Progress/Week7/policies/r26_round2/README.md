# Preserved Round 2 experimental reference

`model_1199.pt` is the 1,200-update Round 2 candidate used as the parent for
Round 4. The adjacent `run.json` preserves the original training metadata;
absolute paths in that historical record need not exist on another computer.
The Week7 evaluator reads the task version and round from this manifest.

This reference passed the reported held-pose forward matrices but retains
toe-in and unequal foot clearance. It is not a hardware release and does not
replace the accepted R25 policy. See the Round 2 and Round 3 result reports.
