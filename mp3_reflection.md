# MP3 Reflection — Grounded Q&A

**Author:** Sushweta Biswas

**Project:** Week 12 — A Cited Document Assistant

## Grounding & abstention

I implemented a grounded question-answering assistant over five
Sherlock Holmes stories using OpenAI embeddings, Qdrant retrieval,
numbered context passages and an evidence-based generation prompt.

The corpus was divided into 73 overlapping chunks. The assistant
retrieves relevant passages, checks whether the top similarity score
meets the retrieval threshold, and instructs the LLM to answer only
from the supplied context.

The assistant correctly abstained from questions about Professor
Moriarty, the boiling point of water and the Baskerville family curse.
These answers were not available in the supplied corpus.

However, my initial evaluation with K=3 and threshold=0.30 revealed
three false abstentions:

- The real identity of Jabez Wilson's assistant.
- The weapon and method used to kill Julia Stoner.
- The theft and hiding place of the Blue Carbuncle.

Interestingly, retrieval hit rate was already 100%. The correct
documents were retrieved, but the specific passages containing
the answers were not always included in the context.

I investigated this further using a question about the real purpose
of the Red-Headed League. The passage explaining the bank robbery
scheme appeared at retrieval rank 7, with a similarity score of 0.456.
The assistant abstained at K=3 and K=5, but answered correctly at
K=7 and K=10, citing passage [7].

This demonstrated that retrieving the correct document does not
necessarily mean retrieving sufficient evidence to answer a question.

I increased the default retrieval depth to K=7 and retained the
abstention threshold at 0.30. This eliminated all three baseline
false abstentions in the final evaluation.

## Citations

I implemented format_context to assign numbered markers to retrieved
chunks and maintain a mapping between each marker and its source
document, title, section and similarity score.

The validate_citations function extracts numbered citation markers
from the generated answer and checks whether they correspond to
available context passages.

The offline selftests confirmed that valid citation markers are
accepted and fabricated markers are flagged.

For example, a marker referring to a passage outside the supplied
context is identified as invalid.

I did not observe fabricated citation markers in the successful
responses from my final evaluation experiments.

My multi-hop question about Irene Adler demonstrated the value of
multiple citations. At K=7 and threshold=0.30, the assistant combined
evidence from two sections:

- The Execution: Holmes used a false fire alarm to discover the
  photograph's hiding place.
- Irene Adler Wins: Adler recognised Holmes and departed with the
  photograph before he could recover it.

The answer cited both passages [2] and [4].

However, during the threshold=0.524 experiment, a similar answer
cited only passage [4], even though it also described the fire alarm.

The automated evaluation still passed this response. This highlighted
a limitation: a valid citation to the correct document does not
guarantee that every individual claim is supported by that citation.

A future improvement would be claim-level citation verification,
rather than checking only marker validity and document identity.

### Additional citation validation experiment

I tested four controlled citation scenarios with two available
source passages.

| Scenario | Valid markers | Invalid markers | Has citation |
|---|---|---|---|
| Valid [1] | [1] | None | True |
| Fabricated [7] | None | [7] | False |
| Mixed [1][9] | [1] | [9] | True |
| Missing citation | None | None | False |

All four tests passed.

The mixed-citation scenario demonstrated that having a valid
citation does not guarantee that all citation markers are valid.
The evaluator must separately check for fabricated markers.

These were controlled synthetic tests, not fabricated citations
observed in live LLM responses.

Results: docs/experiments/citation_validation_results.txt.

## Reading your eval table

I evaluated six answerable questions and three questions that
required abstention, including three questions authored by me.

### Baseline versus final evaluation

| Metric | Baseline K=3 | Final K=7 |
|---|---:|---:|
| Retrieval hit rate | 100% | 100% |
| Grounded response rate | 50% | 100% |
| Abstention accuracy | 100% | 100% |
| Hallucination rate | 0% | 0% |
| Questions passed | 6/9 | 9/9 |

The weakest baseline metric was grounded response rate, at 50%.
Three answerable questions were incorrectly refused despite
retrieving their expected source documents.

Increasing retrieval depth from 3 to 7 improved grounded response
rate by 50 percentage points without reducing abstention accuracy
or increasing the measured hallucination rate.

The final evaluation achieved:

- Retrieval hit rate: 1.0
- Grounded response rate: 1.0
- Abstention accuracy: 1.0
- Hallucination rate: 0.0

All 13 offline selftests also passed.

Although all four final metrics reached their ideal values, the
grounded response metric remains an area for improvement because
it checks citation validity and expected source identity rather
than independently verifying every factual claim.

The evaluation dataset is also small and was used for tuning.
I would expand it with unseen questions, paraphrased questions,
more difficult multi-hop questions and additional out-of-corpus
questions before making broader claims about reliability.

## One surprise

My biggest surprise was that a RAG system could achieve a 100%
retrieval hit rate while correctly answering only half of its
answerable questions according to the grounded response metric.

Initially, I expected retrieving the correct document to be enough.
The Red-Headed League experiment showed that the decisive evidence
could be ranked below the default retrieval cutoff.

Increasing K allowed the model to access the missing evidence and
answer correctly without changing the embedding model, generation
model, corpus or prompt.

This changed my understanding of retrieval quality. Document-level
retrieval success is not the same as passage-level evidence coverage.

I also learned that a faster RAG system is not necessarily a better
one. An overly restrictive threshold can reduce latency simply by
refusing questions that it should have answered.

## Optional stretch — Retrieval depth and threshold tuning

### Retrieval depth experiment

I tested four retrieval depths while keeping the threshold at 0.30.

| K | Grounded rate | Questions passed | Average latency |
|---|---:|---:|---:|
| 3 | 50% | 6/9 | 1843 ms |
| 5 | 83.3% | 8/9 | 2508 ms |
| 7 | 100% | 9/9 | 2131 ms |
| 10 | 100% | 9/9 | 1972 ms |

All four configurations maintained 100% retrieval hit rate,
100% abstention accuracy and 0% measured hallucination rate.

I selected K=7 because it was the smallest tested retrieval depth
that achieved all four ideal evaluation metrics. K=10 provided
no additional measured quality improvement on this dataset.

### Abstention threshold experiment

With K fixed at 7, I tested five threshold values.

| Threshold | Hit rate | Grounded rate | Abstention accuracy | Hallucination rate | Avg latency |
|---|---:|---:|---:|---:|---:|
| 0.30 | 100% | 100% | 100% | 0% | 2034 ms |
| 0.50 | 100% | 100% | 100% | 0% | 1835 ms |
| 0.524 | 100% | 100% | 100% | 0% | 1705 ms |
| 0.55 | 100% | 83.3% | 100% | 0% | 1457 ms |
| 0.60 | 100% | 33.3% | 100% | 0% | 993 ms |

The lowest top similarity score among answerable questions was
0.528. The highest among unanswerable questions was 0.521.

This produced a narrow separation of only 0.007, suggesting a
candidate threshold of approximately 0.524.

Although threshold 0.524 correctly separated all nine questions
at the retrieval gate, I retained 0.30 as the default to reduce
the risk of false abstentions on unseen questions.

At threshold 0.55, the assistant incorrectly refused the Speckled
Band question. At threshold 0.60, four answerable questions were
incorrectly refused.

At threshold 0.30, the retrieval gate rejects the unrelated
boiling-point question, while the LLM's grounding instructions
correctly handle the Moriarty and Baskerville questions.

The selected configuration therefore uses two layers of protection:
a retrieval similarity gate and evidence-based LLM abstention.

Threshold 0.524 could reduce generation calls from eight to six
on this dataset, but requires independent validation before adoption.

My final configuration is K=7 and threshold=0.30, prioritising
answer coverage and grounding over small measured latency gains.

## Implementation notes and limitations

- Models: gpt-4o-mini and text-embedding-3-small.
- Vector store: Qdrant with 1536-dimensional cosine embeddings.
- Corpus: five documents, 73 chunks.
- Chunk size: 500 characters, with 80-character overlap.
- No LangChain, LlamaIndex or additional RAG frameworks.
- Retrieval depth and threshold are configurable through environment
  variables.
- Persistent local Qdrant was used because a remote Qdrant endpoint
  was unavailable in my Vocareum environment. Remote configuration
  remains supported.
- The starter's Qdrant search API produces a deprecation warning
  but continues to work with the installed client version.
- The measured hallucination rate applies only to the designated
  unanswerable questions; it is not a comprehensive factual
  hallucination assessment.
- Latency measurements are single-run averages and may vary
  because of API and network conditions.

Detailed experiment results are preserved in docs/experiments/.
