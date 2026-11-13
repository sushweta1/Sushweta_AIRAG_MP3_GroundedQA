# MP3 Retrieval and Threshold Optimization

## Baseline
- K: 3
- Threshold: 0.30
- Retrieval hit rate: 100%
- Grounded response rate: 50%
- Abstention accuracy: 100%
- Hallucination rate: 0%
- Evaluation pass rate: 6/9

## Retrieval Depth Sweep

| K | Grounded rate | Passed | Avg latency |
|---|---|---|---|
| 3 | 50% | 6/9 | 1843 ms |
| 5 | 83.3% | 8/9 | 2508 ms |
| 7 | 100% | 9/9 | 2131 ms |
| 10 | 100% | 9/9 | 1972 ms |

All configurations maintained 100% retrieval hit
rate, 100% abstention accuracy and 0% measured
hallucination rate.

## Threshold Sweep (K=7)

| Threshold | Grounded rate | Passed | Avg latency |
|---|---|---|---|
| 0.30 | 100% | 9/9 | 2034 ms |
| 0.50 | 100% | 9/9 | 1835 ms |
| 0.524 | 100% | 9/9 | 1705 ms |
| 0.55 | 83.3% | 8/9 | 1457 ms |
| 0.60 | 33.3% | 5/9 | 993 ms |

## Selected Configuration
- K: 7
- Threshold: 0.30

K=7 is the smallest tested retrieval depth that
achieved all four ideal evaluation metrics.

Threshold 0.30 is retained to reduce the risk
of rejecting unseen answerable questions.

Although threshold 0.524 performed well, the
separation between the highest unanswerable
score (0.521) and lowest answerable score
(0.528) was only 0.007.

## Manual Verification
All six answerable responses at the selected
configuration contained their expected key facts.
All three unanswerable questions were refused.

The multi-hop question successfully combined
citations from The Execution and Irene Adler Wins.

## Limitations
- Only nine evaluation questions.
- The same dataset was used for tuning.
- No independent holdout evaluation.
- Document-level citation validation does not
  guarantee claim-level factual support.
- Similarity scores are not calibrated probabilities.
- Latency measurements are subject to API variability.
- Hallucination rate measures unsupported answers
  to designated unanswerable questions only.
