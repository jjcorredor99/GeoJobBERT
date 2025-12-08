from utils.chunks_by_tokens import _chunks_by_tokens

def chunker(
    texts: dict,
    tokenizer,
    chunk_window: int = 250,
    max_length: int = 384,
    chunk_overlap: int = 20,
):
    """
    texts: dict like {"job": job_text, "cand": cand_text}
           or any {"name": text} mapping.

    Returns, for each key `name`:
        f"{name}_chunks_input_ids"
        f"{name}_chunks_attention_mask"

    Todos los valores son listas de ints (JSON-friendly).
    """
    effective_window = max(8, min(chunk_window, max_length - 2))

    out = {}
    for name, text in texts.items():
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        chunks = _chunks_by_tokens(
            ids,
            window=effective_window,
            overlap=chunk_overlap,
            tokenizer=tokenizer,
        )

        # Fuerza todo a listas de ints puras (sin numpy ni tensores)
        chunks_py = [list(map(int, ch)) for ch in chunks]
        masks = [[1] * len(ch) for ch in chunks_py]

        out[f"{name}_chunks_input_ids"] = chunks_py
        out[f"{name}_chunks_attention_mask"] = masks

    # 👉 devolvemos el dict, NO un string JSON
    return out


def chunk_single_text(
    text: str,
    name: str,
    tokenizer,
    chunk_window: int = 250,
    max_length: int = 384,
    chunk_overlap: int = 20,
):
    """Convenience wrapper for a single field."""
    return chunker(
        {name: text},
        tokenizer=tokenizer,
        chunk_window=chunk_window,
        max_length=max_length,
        chunk_overlap=chunk_overlap,
    )

