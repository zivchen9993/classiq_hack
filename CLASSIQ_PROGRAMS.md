# Saved Classiq temporal-antenna programs

These IDs come from the synthesized program artifacts in
`results/classiq_tiny_validation_2026_09_09/`. Classiq's `show(qprog)` function
opens synthesized programs in the circuit visualizer. Login may be required.

- Canonical saved 3-step Suzuki-Trotter DCQO:
  `https://platform.classiq.io/circuit/3J50w5QGnrfsTKrNJfqBY2AwtpD`
- Earlier 1-step Suzuki-Trotter diagnostic:
  `https://platform.classiq.io/circuit/3J50ebY1NLuM9PYwO5flLIku24e`
- Sequential-Pauli 3-step diagnostic (failed numerical validation; do not use as
  the canonical result):
  `https://platform.classiq.io/circuit/3J51e4b3HjrrwQ99ZZ8gCElmMog`

Static repository figures:

- `figures/final_temporal/classiq_logical_circuit.png` - logical block diagram
  derived from saved Classiq metadata; **not a screenshot of the Classiq UI**.
- `figures/final_temporal/classiq_circuit_resources.png` - measured gate counts,
  depth and width from the saved synthesized program.
- `figures/final_temporal/classiq_validation.png` - local-vs-Classiq numerical
  validation result, explicitly showing the unresolved P(optimum) mismatch.

To open the actual UI visualization from the saved JSON in an authenticated SDK
environment:

```bash
python show_classiq_circuit.py \
  results/classiq_tiny_validation_2026_09_09/three_step/classiq_program.json
```
