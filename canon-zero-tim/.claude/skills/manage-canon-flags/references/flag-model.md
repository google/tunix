# Canon flag model

Use this reference when adding, changing, grouping, or retiring flags.

## Taxonomy

| Tier | Meaning | Default posture | Promotion/retirement rule |
|---|---|---|---|
| Numerical | Changes logits, reductions, precision, forward/backward, loss, or optimizer math | off; exact workload opt-in | verify mode, target certification, then deliberate default/weld; welding is a program change |
| Observational/performance | Timing, evidence batching, reports, tracing, observer-only values | off or verify first | prove observer neutrality and required marker before defaulting |
| Diagnostic | Bisection, capture, replay, incident-specific probes | off and case-scoped | retire with the diagnostic case; preserve verdict/evidence |
| Workload/infrastructure | Admission, topology, paths, stages, deadlines, launch identity | explicit and fail-closed | retire with workload or keep as durable launch contract |

The registry in `canon-zero-tim/FLAGS.md` is authoritative. Prefixes do not
determine semantics: a workload-named flag can still alter numerics, and a
parameter does not necessarily activate its associated implementation.

## Parser kinds

Determine the parser from source, not the spelling:

- presence-sensitive: `if "FLAG" in os.environ` or shell `[[ -v FLAG ]]`;
- boolean: exact `0|1` contract;
- enum/mode: one selector with a closed vocabulary;
- numeric parameter: meaningful only with a separately verified switch;
- path/list: empty, missing, and malformed must be distinguished.

For presence-sensitive keys, `FLAG=0` can still activate code. For Docker or
Kubernetes, `-e FLAG=` creates an empty key and prevents
`os.environ.get("FLAG", default)` from using its default. Tests must cover
missing, empty, zero, valid, and invalid values when those states differ.

## Source and process ownership

Record one authoritative writer and one exact reader process. Common writers
are renderer env entries, `_canonical_engine.env`, workload profiles, and
`00_env.sh`. Common readers are the JAX client, vLLM engine process, Pathways
proxy/compiler, sandbox Pod, or postflight shell.

Child-shell `unset` does not clear a parent renderer value. The generated
`env.sh` must be an authoritative managed-namespace snapshot: clear managed
keys, then export the resolved set, without serializing secrets. Test the real
parent reload with stale raw values deliberately seeded.

Pathways compiler flags must be asserted in the proxy/server environment that
compiles HLO. Client visibility is not delivery evidence. Backend flags do not
belong in application `sys.argv`.

## Paired treatments

Use one arm selector as the source of truth and derive every subordinate flag.
Maintain a truth table with exact values and absences. Reject contradictory
manual overrides before initialization.

Observer-only code may share data with a numerical gate, but it must not reuse
the treatment flag if doing so would enable canonical numerical code in the
control arm. Give the observer its own default-off flag, exact signature,
manifest, provenance marker, and negative arm control.

## Registry discipline

Every settable `CANON_*` name needs:

- one appendix entry;
- semantics and tier;
- default and lifecycle;
- objective sunset condition;
- owning workload or cross-workload contract.

Run `scripts/audit_flag_registry.py` after edits. New names in a diff that are
not registered are fatal. An inventory count mismatch or duplicate is fatal;
ordering drift is reported so it can be cleaned deliberately rather than
silently rewriting the registry.
