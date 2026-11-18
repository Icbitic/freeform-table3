"""Fresh free-form generation and joint / five-control head patching."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import argparse
from collections import Counter
import json
import platform
import time

import torch
import transformers

from common import digest, messages, protocol, read_jsonl, run_inputs_digest, write_json
from generation import rollout
from patcher import HeadPatcher
from report import compare, summarize
from scoring import check_replay, shared_answer_prefix_length


def validate_heads(patcher, settings):
    selected = settings["selected_heads"]
    counts = Counter(layer for layer, _ in selected)
    for layer in counts:
        width = patcher.layers[layer].self_attn.num_key_value_groups
        if width != settings["kv_group_widths"][str(layer)]:
            raise ValueError("Model KV grouping differs from the original experiment")
    for members in settings["control_heads"].values():
        if Counter(layer for layer, _ in members) != counts:
            raise ValueError("Control does not match the selected layer counts")
        for layer in counts:
            width = settings["kv_group_widths"][str(layer)]
            chosen = [head // width for l, head in members if l == layer]
            excluded = {head // width for l, head in selected if l == layer}
            if len(set(chosen)) != len(chosen) or set(chosen) & excluded:
                raise ValueError("Control contains excluded or duplicate KV groups")
    for layer, head in selected + [pair for group in settings["control_heads"].values() for pair in group]:
        if not (0 <= layer < len(patcher.layers) and 0 <= head < patcher.n_heads):
            raise ValueError("Head index outside model")


@torch.inference_mode()
def run_experiment(model, tokenizer, device, output_dir, *, model_key,
                   question_limit=None, callbacks=None):
    """Run with ordinary model objects; optional callbacks handle external progress/checkpoints."""
    cfg = protocol()
    settings = cfg["models"][model_key]
    storage = RunStorage(output_dir)
    callbacks = callbacks or {}
    restore = callbacks.get("restore", storage.restore)
    checkpoint_due = callbacks.get("checkpoint_due", storage.checkpoint_due)
    checkpoint = callbacks.get("checkpoint", storage.checkpoint)
    progress = callbacks.get("progress", storage.progress)
    if model.dtype != torch.bfloat16:
        raise ValueError("Use the original bfloat16 weights without quantization")
    if digest(tokenizer.get_chat_template()) != settings["chat_template_sha256"]:
        raise ValueError("Tokenizer/chat template differs from the original run")
    eos = {tokenizer.eos_token_id}
    model_eos = model.generation_config.eos_token_id
    eos.update(model_eos if isinstance(model_eos, (tuple, list)) else [model_eos])
    eos.discard(None)
    if sorted(eos) != settings["eos_ids"]:
        raise ValueError("EOS configuration differs from the original run")
    examples = read_jsonl(ROOT / "data/nq2.jsonl")
    if [e["id"] for e in examples] != cfg["question_ids"]:
        raise ValueError("Question pool changed")
    if question_limit is not None:
        if not 1 <= question_limit <= len(examples):
            raise ValueError("Invalid question limit")
        examples = examples[:question_limit]
    run_inputs_sha256 = run_inputs_digest(cfg["models"])
    checkpoint_run = {"model": model_key, "question_ids": [e["id"] for e in examples]}
    fingerprint = digest({**checkpoint_run, "run_inputs_sha256": run_inputs_sha256})
    state = restore(default={"fingerprint": fingerprint, "rows": [], "identity_checked": False})
    if state["fingerprint"] != fingerprint:
        # Recognize checkpoints from the validated bundle when the run files match.
        validation_path = ROOT / "validation.json"
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
        legacy_fingerprint = digest({
            **checkpoint_run, "checksums": validation.get("validated_experiment_checksums_sha256"),
        })
        if (validation.get("compatible_run_inputs_sha256") != run_inputs_sha256
                or state["fingerprint"] != legacy_fingerprint):
            raise ValueError("Checkpoint belongs to a different experiment")
        state["fingerprint"] = fingerprint
    rows = state["rows"]
    if [r["example_id"] for r in rows] != [e["id"] for e in examples[:len(rows)]]:
        raise ValueError("Checkpoint question coverage is invalid")
    groups = {"decision": settings["selected_heads"],
              **{f"control_{d}": heads for d, heads in settings["control_heads"].items()}}
    training = model.training
    model.eval()
    patcher = HeadPatcher(model, device, tokenizer)
    started = time.monotonic()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        validate_heads(patcher, settings)

        def baseline(prompt, example):
            reply, _ = rollout(patcher, prompt, example, eos, cfg["max_new_tokens"])
            reply["prompt_ids_sha256"] = digest(prompt)
            return reply

        for example in examples[len(rows):]:
            clean_prompt = patcher.encode(messages(example, None, cfg["template"]))
            # Both branches are freshly generated. Reference outputs are not used here.
            clean = baseline(clean_prompt, example)
            donors, attacks = {}, []
            for family in ("logical", "credibility", "emotional"):
                for index, appeal in enumerate(example["persuasions"][family]):
                    prompt = patcher.encode(messages(example, appeal, cfg["template"]))
                    persuaded = baseline(prompt, example)
                    attack = {"persuasion_type": family, "persuasion_index": index, "baseline": persuaded}
                    attacks.append(attack)
                    if clean["outcome"] != "correct" or persuaded["outcome"] != "target":
                        continue
                    clean_onset, attack_onset = clean["answer_onset"], persuaded["answer_onset"]
                    if clean_onset is None or attack_onset is None:
                        attack["skip_reason"] = "ambiguous_token_boundary"
                        continue
                    clean_ids, attack_ids = clean["token_ids"], persuaded["token_ids"]
                    shared = shared_answer_prefix_length(clean_ids[clean_onset:], attack_ids[attack_onset:])
                    clean_cut, attack_cut = clean_onset + shared, attack_onset + shared
                    clean_prefix, attack_prefix = clean_ids[:clean_cut], attack_ids[:attack_cut]
                    attack.update(
                        shared_answer_tokens=shared,
                        matched_lead_in=clean_ids[:clean_onset] == attack_ids[:attack_onset],
                        clean_patch_token_id=(clean_prefix or clean_prompt)[-1],
                        persuaded_patch_token_id=(attack_prefix or prompt)[-1],
                    )
                    if clean_cut not in donors:
                        replay, donor = rollout(patcher, clean_prompt, example, eos, cfg["max_new_tokens"],
                                                lead_in=clean_prefix, capture=True)
                        check_replay(clean, replay)
                        donors[clean_cut] = donor
                    next_tokens = ()
                    if clean_cut < len(clean_ids) and attack_cut < len(attack_ids):
                        next_tokens = (clean_ids[clean_cut], attack_ids[attack_cut])
                    kwargs = {"lead_in": attack_prefix, "answer_token_ids": next_tokens}
                    replay, recipient = rollout(patcher, prompt, example, eos, cfg["max_new_tokens"],
                                                capture=not state["identity_checked"], **kwargs)
                    check_replay(persuaded, replay)
                    if not state["identity_checked"]:
                        for members in groups.values():
                            identity, _ = rollout(patcher, prompt, example, eos, cfg["max_new_tokens"],
                                                  intervention=(members, recipient), **kwargs)
                            check_replay(replay, identity)
                            if identity["onset_scores"] != replay["onset_scores"]:
                                raise ValueError("Identity patch changed next-token scores")
                        state["identity_checked"] = True
                    attack["arms"] = {
                        arm: rollout(patcher, prompt, example, eos, cfg["max_new_tokens"],
                                     intervention=(members, donors[clean_cut]), **kwargs)[0]
                        for arm, members in groups.items()
                    }
            rows.append({"example_id": example["id"], "clean": clean, "attacks": attacks})
            progress(len(rows), len(examples))
            if checkpoint_due():
                checkpoint(state)
        complete = len(rows) == 131
        summary = summarize(rows, settings, {e["id"]: e for e in examples}, complete=complete)
        summary["identity_checked"] = state["identity_checked"]
        observed = {key: summary[key] for key in settings["expected"]}
        summary["published_counts_match"] = complete and observed == settings["expected"]
        summary["published_percentages_match"] = complete and (
            f"{summary['decision_percent']:.1f}", f"{summary['control_mean_percent']:.1f}"
        ) == (
            f"{100 * settings['expected']['decision_correct'] / settings['expected']['eligible']:.1f}",
            f"{100 * sum(settings['expected']['control_correct']) / (5 * settings['expected']['eligible']):.1f}"
        )
        # Only after completing generation do we compare against the historical outputs.
        comparison = compare(rows, read_jsonl(ROOT / "reference" / f"{model_key}.jsonl"))
        runtime = {
            "model_key": model_key,
            "repo_id": settings["repo_id"], "revision": settings["revision"],
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "model_class": type(model).__name__, "dtype": str(model.dtype),
            "attention_implementation": getattr(patcher.config, "_attn_implementation", None),
            "device": str(device), "run_inputs_sha256": run_inputs_sha256,
            "seconds_this_attempt": time.monotonic() - started,
        }
        if torch.device(device).type == "cuda":
            runtime.update(gpu=torch.cuda.get_device_name(device),
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "rollouts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        for name, value in (("summary.json", summary), ("comparison.json", comparison), ("runtime.json", runtime)):
            write_json(output / name, value)
        print(json.dumps({"model": model_key, **summary}), flush=True)
        return summary
    finally:
        patcher.close()
        model.train(training)


class RunStorage:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = time.monotonic()

    def restore(self, default):
        path = self.output / "checkpoint.json"
        return json.loads(path.read_text()) if path.exists() else default

    def checkpoint_due(self):
        return time.monotonic() - self.last_checkpoint > 90

    def checkpoint(self, state):
        path = self.output / "checkpoint.json"
        temporary = path.with_suffix(".tmp")
        write_json(temporary, state)
        temporary.replace(path)
        self.last_checkpoint = time.monotonic()

    def progress(self, current, total):
        print(f"{current}/{total} questions", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(protocol()["models"]))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", help="Optional local copy of the pinned Hugging Face snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--question-limit", type=int, help="Smoke run only; omit for the full table")
    args = parser.parse_args()
    settings = protocol()["models"][args.model]
    source = args.model_path or settings["repo_id"]
    kwargs = {"local_files_only": True} if args.model_path else {"revision": settings["revision"]}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
    loader = transformers.Gemma3ForConditionalGeneration if args.model == "gemma" else transformers.AutoModelForCausalLM
    model = loader.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
    model.to(args.device)
    run_experiment(model, tokenizer, torch.device(args.device), args.output,
                   model_key=args.model, question_limit=args.question_limit)


if __name__ == "__main__":
    main()
