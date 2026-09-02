"""PG-19 long-form generation task loader."""
from __future__ import annotations
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import TaskSpec, AgentInput


def load_pg19_tasks(
    n_books: int = 20,
    min_tokens: int = 8192,
    prefix_tokens: int = 512,
    generation_lengths: List[int] = None,
) -> List[TaskSpec]:
    """
    Load PG-19 test split for long-form generation.

    Args:
        n_books: Number of books to use (sorted by book_id)
        min_tokens: Minimum length in tokens
        prefix_tokens: Number of prefix tokens to provide
        generation_lengths: List of generation lengths [2048, 4096, 8192]

    Returns:
        List of TaskSpec objects, one per (book, length) pair
    """
    if generation_lengths is None:
        generation_lengths = [2048, 4096, 8192]

    # Load PG-19 test split
    print("Loading PG-19 test split...")
    ds = load_dataset("emozilla/pg19", split="test")

    # Filter by length and sort by book_id
    candidates = []
    for idx, item in enumerate(ds):
        book_id = item.get("book_id", idx)
        text = item.get("text", "")
        # Rough token count (word-based approximation)
        token_count = len(text.split())
        if token_count >= min_tokens:
            candidates.append({
                "book_id": book_id,
                "text": text,
                "token_count": token_count,
            })

    # Sort by book_id and take first n_books
    candidates.sort(key=lambda x: x["book_id"])
    candidates = candidates[:n_books]

    print(f"Selected {len(candidates)} books with >= {min_tokens} tokens")

    # Create tasks
    tasks = []
    for book in candidates:
        book_id = book["book_id"]
        text = book["text"]

        # Split into prefix and continuation
        words = text.split()
        prefix = " ".join(words[:prefix_tokens])
        continuation = " ".join(words[prefix_tokens:])

        for gen_length in generation_lengths:
            # Truncate continuation to desired length
            target_text = " ".join(words[prefix_tokens:prefix_tokens + gen_length])

            # Create task
            task = TaskSpec(
                task_id=f"pg19_book{book_id}_len{gen_length}",
                task_name="pg19",
                agents=[AgentInput(role="writer", text=prefix)],
                query=f"Continue the text for {gen_length} tokens.",
                gold_answer=target_text,
                metadata={
                    "book_id": book_id,
                    "prefix_tokens": prefix_tokens,
                    "generation_length": gen_length,
                    "total_tokens": book["token_count"],
                }
            )
            tasks.append(task)

    print(f"Created {len(tasks)} tasks ({len(candidates)} books × {len(generation_lengths)} lengths)")
    return tasks


def compute_repetition_4gram(text: str) -> float:
    """Compute 4-gram repetition ratio."""
    words = text.split()
    if len(words) < 4:
        return 0.0

    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))


def compute_entity_consistency(generated: str, reference: str) -> float:
    """
    Compute entity consistency between generated and reference text.
    Simple heuristic: check if named entities in reference appear in generated.
    """
    # Extract capitalized words as proxy for entities
    def extract_entities(text):
        words = text.split()
        entities = set()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 1:
                # Check if it's likely a named entity (not start of sentence)
                if i > 0 and words[i-1] not in ['.', '!', '?']:
                    entities.add(word)
        return entities

    ref_entities = extract_entities(reference)
    gen_entities = extract_entities(generated)

    if not ref_entities:
        return 1.0

    # Check overlap
    overlap = len(ref_entities & gen_entities)
    return overlap / len(ref_entities)


def compute_coherence_per_block(text: str, block_size: int = 512) -> List[float]:
    """
    Compute coherence score for each 512-token block.
    Simple heuristic: average sentence length and punctuation density.
    """
    words = text.split()
    scores = []

    for i in range(0, len(words), block_size):
        block = " ".join(words[i:i+block_size])
        if not block:
            continue

        # Sentence count (approximate)
        sentences = block.count('.') + block.count('!') + block.count('?')
        if sentences == 0:
            sentences = 1

        # Average sentence length
        avg_sent_len = len(block.split()) / sentences

        # Punctuation density
        punct_count = sum(1 for c in block if c in '.,!?;:')
        punct_density = punct_count / len(block)

        # Coherence score (heuristic)
        score = min(avg_sent_len / 20.0, 1.0) * 0.5 + min(punct_density * 10, 1.0) * 0.5
        scores.append(score)

    return scores


def evaluate_pg19_generation(
    generated: str,
    reference: str,
    prefix: str,
) -> Dict[str, Any]:
    """
    Evaluate PG-19 generation with multiple metrics.

    Args:
        generated: Generated text
        reference: Ground truth continuation
        prefix: Input prefix

    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    gen_length = len(generated.split())
    ref_length = len(reference.split())

    # Repetition
    rep_4gram = compute_repetition_4gram(generated)

    # Entity consistency
    entity_consistency = compute_entity_consistency(generated, reference)

    # Coherence per block
    coherence_scores = compute_coherence_per_block(generated)
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    # Topic retention (simple: check if key words from prefix appear in generated)
    prefix_words = set(prefix.lower().split())
    gen_words = set(generated.lower().split())
    topic_retention = len(prefix_words & gen_words) / len(prefix_words) if prefix_words else 0.0

    # First EOS position (if applicable)
    first_eos = generated.find('<|endoftext|>')
    if first_eos == -1:
        first_eos = len(generated)

    return {
        "generation_length": gen_length,
        "reference_length": ref_length,
        "repetition_4gram": rep_4gram,
        "entity_consistency": entity_consistency,
        "avg_coherence": avg_coherence,
        "topic_retention": topic_retention,
        "first_eos_position": first_eos,
        "coherence_per_block": coherence_scores,
    }
