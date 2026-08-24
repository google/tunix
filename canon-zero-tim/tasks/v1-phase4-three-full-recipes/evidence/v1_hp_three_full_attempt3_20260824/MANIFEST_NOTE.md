# Attempt 3 artifact manifest correction

Operator commit `65606a985aa869f09a3bd3a39a3c9268a432aa71` appended a
`SHA256SUMS` entry for `SHA256SUMS` itself. That recorded digest describes an
earlier version of the file and cannot validate after the self-entry is added.
The original manifest is preserved unchanged as immutable evidence of that
packaging error.

Use the additive non-self-referential `SHA256SUMS.artifacts` manifest to verify
the three raw logs and `receipt.json`:

```bash
sha256sum -c SHA256SUMS.artifacts
```

All four entries pass. This correction changes no raw log, receipt, failure
classification, numerical claim, or target result.
