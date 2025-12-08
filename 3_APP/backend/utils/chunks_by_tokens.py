def _chunks_by_tokens(input_ids, window, overlap, tokenizer):
    # window incluye ya tokens especiales? NO. Trabajamos sin especiales y los añadimos por chunk.
    if window <= 0:
        raise ValueError("chunk_window debe ser > 0")
    if overlap < 0:
        overlap = 0

    # Para evitar chunks vacíos o stride 0
    stride = max(1, window - overlap)
    chunks = []
    for start in range(0, len(input_ids), stride):
        end = start + window
        piece = input_ids[start:end]
        if len(piece) == 0:
            continue
        piece = tokenizer.build_inputs_with_special_tokens(piece)
        chunks.append(piece)
        if end >= len(input_ids):
            break
    return chunks