"""Portable inputs and prompt rendering; no GPU dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return json.loads((ROOT / "protocol.json").read_text())


def run_inputs_digest(model_keys):
    paths = (
        "common.py", "data/nq2.jsonl", "generation.py", "patcher.py",
        "protocol.json", "report.py", "requirements.txt", "run.py", "scoring.py",
        *(f"reference/{key}.jsonl" for key in model_keys),
    )
    return digest({name: file_hash(ROOT / name) for name in paths})


def messages(example, appeal, fields):
    """The original renderer with its choice-free branch retained."""
    first = f"{fields['opening']}\n\nQuestion:\n{example['question']}"
    if appeal is None:
        framing = fields["clean"]
    else:
        if not appeal.strip():
            raise ValueError("Empty appeal")
        parts = [fields["persuasive_prefix"]]
        if fields.get("persuasion_open", "<my_persuasion>"):
            parts.append(fields.get("persuasion_open", "<my_persuasion>"))
        parts.append(appeal.strip())
        if fields.get("persuasion_close", "</my_persuasion>"):
            parts.append(fields.get("persuasion_close", "</my_persuasion>"))
        parts.append(fields["persuasive_suffix"])
        framing = "\n\n".join(parts)
    return [
        {"role": "system", "content": fields["system"]},
        {"role": "user", "content": first},
        {"role": "assistant", "content": fields["acknowledgement"]},
        {"role": "user", "content": f"{framing}\n\n{fields['output_instruction']}"},
    ]
