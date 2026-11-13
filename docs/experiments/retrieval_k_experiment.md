# MP3 Retrieval Depth Experiment

## Objective
Investigate why an answerable question resulted in abstention.

## Question
What was the real purpose behind the creation of the
Red-Headed League, and why did the criminals need
Jabez Wilson to leave his pawnshop?

## Controlled Configuration
- Embedding model: text-embedding-3-small
- Chat model: gpt-4o-mini
- Corpus: 5 documents, 73 chunks
- Chunk size: 500
- Chunk overlap: 80
- Similarity metric: Cosine
- Abstention threshold: 0.30
- Vector store: Persistent local Qdrant

## Results

| K | Evidence retrieved | Abstained | Correct answer | Latency |
|---|---|---|---|---|
| 3 | No | Yes | No | 2110 ms |
| 5 | No | Yes | No | 4816 ms |
| 7 | Yes | No | Yes | 4046 ms |
| 10 | Yes | No | Yes | 3874 ms |

## Root Cause
The correct document was retrieved at every K value.
However, the passage containing the direct answer
was ranked seventh, with a similarity score of 0.456.

## Findings
Increasing K from 3 to 7 recovered the missing
evidence and eliminated false abstention for this
question. Both K=7 and K=10 cited passage [7]
without invalid citations.

## Conclusion
Document-level retrieval success does not guarantee
passage-level relevance or answer completeness.
Retrieval depth can affect task success even when
the embedding model and abstention threshold
remain unchanged.

K=7 was sufficient for this question. Further
evaluation is required before changing the default K.

Latency measurements are from single runs and
should not be interpreted as general benchmarks.
