"""Teacher-forced trajectory linearization — the load-bearing module.

Turns a `Trajectory` (tool calls + thoughts + answer) into one token
sequence with per-position segment IDs so the whole ReAct chain can be
scored by a single forward pass.
"""
