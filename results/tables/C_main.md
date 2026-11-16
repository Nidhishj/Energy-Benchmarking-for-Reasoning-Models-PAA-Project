# Experiment C: cutting thinking short (Qwen3-8B, MATH-500)

Run `20261116-145354_C_gpu1_ba35d077` (status complete), source traces `20261019-122926_B2b_gpu0_feea8f05`, GPU NVIDIA RTX 6000 Ada Generation, idle 45.5 W. Forced end of thinking: `'\n</think>\n\n'`; answer cap 2048 tokens.
Traces: 81 complete (28 problems); 9 source traces skipped (hit the token cap, no natural end).

Thinking off: `20261116-185413_B2b_expc_off_gpu1_18d69d50`, matched on (problem, trial): 81 rows, session `20261116-185413_server_gpu1`.

| Budget | Traces cut | Thinking tok | Answer tok | Accuracy (95% range) | J/query | J/query idle-sub | J/correct | Energy saved vs full | Right→wrong | Wrong→right | Answer hit cap | Reopened thinking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off |  | 0 | 534 | 95.1% (88–98) | 3440 | 2671 | 3619 | +85% |  |  |  |  |
| p25 | 81 | 707 | 511 | 97.5% (91–99) | 7947 | 6539 | 8149 | +64% | 2 | 0 | 1 | 0 |
| p50 | 81 | 1414 | 531 | 100.0% (95–100) | 12757 | 10514 | 12757 | +43% | 0 | 0 | 0 | 0 |
| p75 | 81 | 2121 | 511 | 100.0% (95–100) | 17331 | 14301 | 17331 | +22% | 0 | 0 | 0 | 0 |
| full |  | 2828 | 552 | 100.0% (95–100) | 22357 | 18467 | 22357 | +0% |  |  |  |  |
| t512 | 81 | 512 | 575 | 95.1% (88–98) | 7090 | 5831 | 7459 | +68% | 4 | 0 | 3 | 1 |
| t1024 | 81 | 1024 | 544 | 96.3% (90–99) | 10203 | 8394 | 10596 | +54% | 3 | 0 | 1 | 0 |
| t2048 | 45 | 1817 | 545 | 97.5% (91–99) | 15470 | 12749 | 15862 | +31% | 2 | 0 | 0 | 0 |
| t4096 | 14 | 2461 | 535 | 100.0% (95–100) | 19749 | 16300 | 19749 | +12% | 0 | 0 | 0 | 0 |

## By MATH level (percentage budgets): accuracy / J per correct answer

| Level | Traces | p25 | p50 | p75 | full | off |
|---|---|---|---|---|---|---|
| 1 | 18 | 100% / 5244 J | 100% / 8832 J | 100% / 11792 J | 100% / 15180 J | 100% / 1647 J |
| 2 | 17 | 94% / 9132 J | 100% / 14447 J | 100% / 20032 J | 100% / 26014 J | 82% / 4502 J |
| 3 | 18 | 94% / 6300 J | 100% / 9273 J | 100% / 12268 J | 100% / 15783 J | 100% / 2979 J |
| 4 | 14 | 100% / 8149 J | 100% / 13346 J | 100% / 17705 J | 100% / 22730 J | 100% / 3739 J |
| 5 | 14 | 100% / 13004 J | 100% / 19642 J | 100% / 27305 J | 100% / 35222 J | 93% / 6154 J |

Notes:
- Thinking energy comes from a teacher-forced replay of the saved tokens (one forward per token); in a test it ran ~2% faster than generate() at the same power, so thinking energy is ~2% low.
- Absolute budgets: traces whose thinking ends before k are not cut (column 'Traces cut'); they use their full row.
- J/correct = total energy of all queries at that budget ÷ number of correct answers.
- Skipped token-capped traces are the longest ones, mostly on hard problems; accuracy here is for problems the model finishes thinking on.
