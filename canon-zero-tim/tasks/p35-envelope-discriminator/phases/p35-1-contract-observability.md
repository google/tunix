# P35.1 contract observability

Status: completed

## Question

Can every r18 value-boundary report state exactly how many action elements and bytes differ,
without conflating full-array hashes with action-only equality?

## Single variable

Only reporting and fail-closed validation change. Model execution, precision, scheduler,
sampling, loss, gradients and optimizer state are untouched.

## Required measurements

For every boundary:

- validity of shape/dtype/mask contract;
- differing and total bytes;
- differing and total action elements;
- byte and element fractions;
- first mismatching action element;
- maximum absolute difference as a descriptive value;
- full-array hashes and action-masked hashes as separately named fields.

## Controls

- Exact positive control: every count is zero and masked hashes match.
- One-ULP negative control: exactly one action element differs.
- Signed-zero negative control: numerical equality must not pass bitwise equality.
- Mask control: a drift outside the action mask changes full hashes but not the action boundary.

## Exit gate

The focused unit suite passes, each negative control is observed, and no existing report consumer
loses the legacy `differing_bytes` field.

Result: PASS on 2026-08-09. Exact-image alignment tests 13/13 and classifier tests 5/5 passed.
