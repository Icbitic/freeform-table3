"""Original one-position head interchange and greedy continuation."""
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from scoring import classify_reply

@contextmanager
def replace_onset_heads(patcher, members, donor):
    """Replace selected head outputs at their layers in one recipient forward."""
    by_layer = {}
    for layer, head in members:
        by_layer.setdefault(layer, []).append(head)

    def replace(layer, heads, module, args):
        value = args[0]
        if value.shape[:2] != (1, 1):
            raise ValueError("Expected only the answer-onset query")
        value = value.clone()
        for head in heads:
            start = head * patcher.head_dim
            stop = start + patcher.head_dim
            value[0, -1, start:stop] = donor.head[layer][0, -1, start:stop]
        return value, *args[1:]

    with ExitStack() as hooks:
        for layer, heads in by_layer.items():
            projection = patcher.layers[layer].self_attn.o_proj
            handle = projection.register_forward_pre_hook(partial(replace, layer, heads))
            hooks.callback(handle.remove)
        yield


def rollout(patcher, prompt, example, eos_ids, budget, *, lead_in=(), intervention=None,
            capture=False, answer_token_ids=(), cache=None):
    """Replay natural lead-in IDs and keep the intervened KV cache for decoding."""
    ids = list(prompt)
    if cache is None:
        cache = patcher.prefill(ids)
    for expected in lead_in:
        logits, _ = patcher.last(ids, cache)
        if logits.argmax().item() != expected:
            raise ValueError("Unpatched lead-in did not replay")
        ids.append(expected)

    with replace_onset_heads(patcher, *intervention) if intervention else nullcontext():
        logits, activations = patcher.last(ids, cache, capture=capture)
    onset_scores = None
    if answer_token_ids:
        selected = logits[0, list(answer_token_ids)]
        onset_scores = {
            "token_ids": list(answer_token_ids),
            "logits": selected.tolist(),
            "probabilities": (selected - logits[0].logsumexp(dim=-1)).exp().tolist(),
        }

    reply = list(lead_in)
    stopped = False
    while len(reply) < budget:
        token = int(logits.argmax().item())
        if token in eos_ids:
            stopped = True
            break
        reply.append(token)
        ids.append(token)
        if len(reply) < budget:
            logits, _ = patcher.last(ids, cache)
    return {
        **classify_reply(patcher.tokenizer, reply, example, stopped),
        "token_ids": reply,
        "onset_scores": onset_scores,
    }, activations
