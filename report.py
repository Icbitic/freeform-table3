"""Recompute Table 3 from individual replies using only the Python standard library."""
import argparse
from collections import Counter
import json
from pathlib import Path

from common import ROOT, protocol, read_jsonl
from scoring import answer_span, shared_answer_prefix_length


def check_score(reply, example):
    if reply["outcome"] == "truncated":
        if len(reply["token_ids"]) != 32:
            raise ValueError("Truncation was recorded before the token limit")
        return
    span = answer_span(reply["text"], example["choices"])
    expected = "unresolved" if span is None else {
        example["correct_index"]: "correct", example["target_index"]: "target"
    }.get(span[0], "other_choice")
    if reply["outcome"] != expected:
        raise ValueError(f"Reply scoring mismatch: {reply['text']!r}")
    if span is not None and reply.get("choice_index") != span[0]:
        raise ValueError("Choice index mismatch")


def summarize(rows, settings, examples, *, complete=True):
    cfg = protocol()
    if complete and [r["example_id"] for r in rows] != cfg["question_ids"]:
        raise ValueError("Missing, duplicate, or reordered questions")
    counts = Counter()
    matched = 0
    for row in rows:
        example = examples[row["example_id"]]
        check_score(row["clean"], example)
        counts["clean_correct"] += row["clean"]["outcome"] == "correct"
        expected_appeals = [(family, i) for family in ("logical", "credibility", "emotional") for i in range(3)]
        if [(a["persuasion_type"], a["persuasion_index"]) for a in row["attacks"]] != expected_appeals:
            raise ValueError("Expected all nine appeals in their original order")
        for attack in row["attacks"]:
            clean, baseline = row["clean"], attack["baseline"]
            check_score(baseline, example)
            eligible = clean["outcome"] == "correct" and baseline["outcome"] == "target"
            eligible = eligible and clean["answer_onset"] is not None and baseline["answer_onset"] is not None
            if eligible != ("arms" in attack):
                raise ValueError("Patching eligibility differs from clean-correct to target switches")
            if not eligible:
                continue
            shared = shared_answer_prefix_length(clean["token_ids"][clean["answer_onset"]:],
                                                 baseline["token_ids"][baseline["answer_onset"]:])
            if attack["shared_answer_tokens"] != shared:
                raise ValueError("Incorrect first-divergence boundary")
            identical = clean["token_ids"][:clean["answer_onset"]] == baseline["token_ids"][:baseline["answer_onset"]]
            if identical != attack["matched_lead_in"]:
                raise ValueError("Incorrect lead-in metadata")
            matched += identical
            for reply, name in ((clean, "clean_patch_token_id"), (baseline, "persuaded_patch_token_id")):
                cut = reply["answer_onset"] + shared
                if cut and reply["token_ids"][cut - 1] != attack[name]:
                    raise ValueError("Patch is not immediately before the differing answer token")
            if set(attack["arms"]) != {"decision", *[f"control_{i}" for i in range(1, 6)]}:
                raise ValueError("Missing selected-head or control intervention")
            counts["eligible"] += 1
            for arm, reply in attack["arms"].items():
                check_score(reply, example)
                counts[arm] += reply["outcome"] == "correct"
    n = counts["eligible"]
    return {
        "questions": len(rows), "clean_correct": counts["clean_correct"], "eligible": n,
        "identical_lead_in_pairs": matched, "decision_correct": counts["decision"],
        "control_correct": [counts[f"control_{i}"] for i in range(1, 6)],
        "decision_percent": 100 * counts["decision"] / n if n else None,
        "control_mean_percent": 100 * sum(counts[f"control_{i}"] for i in range(1, 6)) / (5 * n) if n else None,
    }


def compare(rows, reference):
    """Report drift; never force fresh generations to match saved continuations."""
    old = {r["example_id"]: r for r in reference}
    counters, details = Counter(), []
    for row in rows:
        prior = old[row["example_id"]]
        pairs = [("clean", row["clean"], prior["clean"])]
        for attack, previous in zip(row["attacks"], prior["attacks"], strict=True):
            label = f"{attack['persuasion_type']}:{attack['persuasion_index']}"
            pairs.append((label, attack["baseline"], previous["baseline"]))
            if ("arms" in attack) != ("arms" in previous):
                counters["eligibility_changes"] += 1
            for arm in set(attack.get("arms", {})) & set(previous.get("arms", {})):
                pairs.append((f"{label}:{arm}", attack["arms"][arm], previous["arms"][arm]))
        for label, actual, expected in pairs:
            counters["replies_compared"] += 1
            if actual["token_ids"] != expected["token_ids"]:
                counters["token_sequence_changes"] += 1
                details.append({"example_id": row["example_id"], "condition": label,
                                "original": expected["text"], "rerun": actual["text"],
                                "original_outcome": expected["outcome"], "rerun_outcome": actual["outcome"]})
            counters["outcome_changes"] += actual["outcome"] != expected["outcome"]
    return {**dict(counters), "changed_replies": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Fresh output root containing MODEL/rollouts.jsonl")
    parser.add_argument("--assert-table", action="store_true", help="Fail unless counts match the published experiment")
    args = parser.parse_args()
    cfg = protocol()
    examples = {e["id"]: e for e in read_jsonl(ROOT / "data/nq2.jsonl")}
    print("| Model | Decision heads | Random heads (mean of five) |")
    print("|---|---:|---:|")
    mismatches = []
    for key, settings in cfg["models"].items():
        path = args.results / key / "rollouts.jsonl" if args.results else ROOT / "reference" / f"{key}.jsonl"
        if args.results and not path.exists():
            path = args.results / key / "results" / "rollouts.jsonl"
        result = summarize(read_jsonl(path), settings, examples)
        print(f"| {settings['label']} | {result['decision_percent']:.1f}% | {result['control_mean_percent']:.1f}% |")
        observed = {name: result[name] for name in settings["expected"]}
        if observed != settings["expected"]:
            mismatches.append({"model": key, "expected": settings["expected"], "observed": observed})
    if mismatches:
        print(json.dumps({"count_differences": mismatches}, indent=2))
        if args.assert_table:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
