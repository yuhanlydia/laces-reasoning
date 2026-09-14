from laces_posttrain.math_verify import extract_final_answer, canonical_numeric, verify_answer


def test_gsm8k_extracts_hash_answer_with_commas_and_sign():
    assert extract_final_answer('work\n#### -1,250') == '-1,250'
    assert canonical_numeric('-1,250') == '-1250'
    assert verify_answer('reasoning\n#### -1,250', '-1250', 'gsm8k')


def test_decimal_numeric_canonicalization_is_exact():
    assert canonical_numeric('1.50') == '3/2'
    assert canonical_numeric('+0.125') == '1/8'
    assert verify_answer('Final answer: 1.5', '3/2', 'gsm8k')


def test_math_boxed_fraction_is_compared_as_reduced_fraction():
    assert extract_final_answer(r'So the result is \\boxed{6/8}.') == '6/8'
    assert canonical_numeric(r'\\frac{3}{4}') == '3/4'
    assert verify_answer(r'We conclude \\boxed{6/8}.', r'\\frac{3}{4}', 'math')


def test_final_answer_anchoring_prefers_last_explicit_answer():
    text = r'First guess: \\boxed{2}. Recheck. Final answer: 3'
    assert extract_final_answer(text) == '3'
    assert verify_answer(text, '3', 'math')
    assert not verify_answer(text, '2', 'math')


def test_prose_substring_is_not_accepted_as_answer():
    assert extract_final_answer('The number 42 appears in the prompt but no final answer is stated.') is None
    assert not verify_answer('The number 42 appears in the prompt.', '42', 'gsm8k')


def test_math_symbolic_matching_is_conservative_not_algebraic_solver():
    assert verify_answer(r'Final answer: x+1', 'x+1', 'math')
    assert not verify_answer(r'Final answer: 1+x', 'x+1', 'math')


def test_invalid_or_unrecognized_answer_returns_false():
    assert canonical_numeric('approximately two') is None
    assert not verify_answer('Final answer: approximately two', '2', 'gsm8k')
