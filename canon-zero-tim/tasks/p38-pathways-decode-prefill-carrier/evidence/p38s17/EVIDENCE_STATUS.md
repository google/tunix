# P38s17 evidence status

This directory is analysis-level evidence recovered from live snapshot 58.
It is not a terminal admitted bundle: `COLLECTED.json` and `COMPLETE.json` are
absent.

The original committed KV classification was not reproducible from the files
in this directory. Re-running the classifier against the six observer records
and the three immutable round capsules gives
`live_kv_fingerprint_equal_on_red_row`: all valid aggregate and sample cells
are equal, and row 255 joins a red coordinate in every round. Differences in
unused page tails are outside the valid extents and are masked by contract.

The corrected classification records the classifier, observer, capsule, and
valid-extent provenance. `SHA256SUMS` excludes itself and seals every file in
this analysis directory. These repairs make the offline conclusion
reproducible; they do not manufacture missing terminal markers.
