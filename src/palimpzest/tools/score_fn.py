from nltk.translate.bleu_score import sentence_bleu

def bleu_scorer(computed_value: str, expected_value: str, max_ngram: int = 3) -> float:
    """
    Calculate BLEU score between computed and expected strings using NLTK.
    
    Args:
        computed_value: The generated/computed text
        expected_value: The reference/expected text
        max_ngram: Maximum n-gram size to consider (default: 3)
    
    Returns:
        float: BLEU score between 0 and 1
    """
    computed_value = str(computed_value)
    expected_value = str(expected_value)

    if not computed_value or not expected_value:
        return 0.0
    
    if computed_value == expected_value:
        return 1.0

    # Simple whitespace tokenization
    reference = [expected_value.split()]  # NLTK expects a list of references
    hypothesis = computed_value.split()
    
    # Calculate weights based on max_ngram
    weights = [1/max_ngram] * max_ngram
    
    return sentence_bleu(reference, hypothesis, weights=weights)