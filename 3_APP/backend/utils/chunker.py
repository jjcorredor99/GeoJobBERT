from .chunks_by_tokens import _chunks_by_tokens


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

    Each is a list[list[int]] of token ids for each chunk.
    """
    out = {}

    for name, text in texts.items():
        if text is None:
            text = ""
        text = str(text)

        # Tokenize without special tokens
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        chunks_ids = _chunks_by_tokens(
            input_ids=tokens,
            window=chunk_window,
            overlap=chunk_overlap,
            tokenizer=tokenizer,
        )

        # Truncate each chunk to max_length
        chunks_ids = [ids[:max_length] for ids in chunks_ids]

        out[f"{name}_chunks_input_ids"] = chunks_ids

        # Attention mask = 1 where there is a token
        chunks_mask = [[1] * len(ids) for ids in chunks_ids]
        out[f"{name}_chunks_attention_mask"] = chunks_mask

    return out


def chunker_single(
    name: str,
    text: str,
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
