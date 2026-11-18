# Fresh-run validation

The packaged experiment was rerun from scratch for all four models, using all
131 questions and all nine appeals per question. Both clean and persuasive
answers were generated independently. The saved reference answers were read
only after generation, for comparison.

Every published percentage and every underlying recovery count matched exactly:

| Model | Decision heads | Random heads, mean of five | Decision recoveries | Control recoveries, draws 1–5 |
|---|---:|---:|---:|---|
| Llama-3 8B | 40.5% | 6.4% | 68/168 | 8, 16, 8, 15, 7 |
| Gemma-3 12B | 23.8% | 1.8% | 29/122 | 0, 4, 2, 0, 5 |
| OLMo-2 7B | 35.5% | 1.4% | 54/152 | 2, 1, 5, 2, 1 |
| Qwen3 4B | 19.6% | 3.0% | 27/138 | 4, 2, 6, 4, 5 |

All 8,720 compared replies matched the reference token sequences, decoded text,
answer-onset positions, and outcome classifications. All 5,240 baseline prompt
hashes and all 580 eligible intervention boundaries also matched. Identity
patches passed for the decision-head set and all five control sets in every
model. The random controls were sampled once with the recorded seeds and
reused across examples.

Validation used Python 3.11.15, PyTorch 2.11.0 with CUDA 12.8, Transformers
4.57.6, bfloat16 weights, SDPA attention, and NVIDIA A800 GPUs. Exact model
revisions and chat-template hashes are pinned in `protocol.json`.

`validation.json` contains the per-model results and runtime information. It
records the digest of the file manifest used for those runs and the digest of
the run files in this archive that can resume checkpoints from those runs.

These results establish exact reproduction in the recorded environment. The
runner reports discrepancies rather than replacing fresh outputs with saved
answers if another environment produces different greedy decisions.
