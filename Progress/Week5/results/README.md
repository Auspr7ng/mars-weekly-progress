# Xbox evidence

`run_verification.json` is the saved bounded snapshot from the final single-left-stick session. `dual_stick_initial_verification.json` documents the earlier layout before the user requested single-stick walking. It is historical evidence, not the current interface.

`input_contract_tests.json` records 17 passing command and Week4 mapping tests. Physical input checks are separate from unit tests. `controller_diagnostics.json` records the successful desktop SDL enumeration alongside failed XInput probes.

The published JSONL files contain actual controller and native simulation samples bounded to the final single-left-stick snapshot. Longer sessions and earlier raw layout revisions remain in the local archive. `summarize_run.py` detects the latest simulated-time reset to select an episode. Running the live demo later appends new data, so save an archival copy before replacing the recorded session.

The trace's requested command and measured velocity have different meanings. Requested limits do not bound instantaneous measured velocity. The frame named `focused` in native SDL packets is an enable field set true by the native reader; it is not a measurement of browser focus.
