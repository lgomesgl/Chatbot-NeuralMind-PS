from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

def rouge_score(generated_response: str, reference_response: str):
    """
    Calculates ROUGE scores between the generated response and the reference response.

    Args:
        generated_response (str): The generated response.
        reference_response (str): The reference response.

    Returns:
        Dict[str, Dict[str, float]]: ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L).
    """
    rouge = Rouge()
    scores = rouge.get_scores(generated_response, reference_response)
    return scores[0]

def bleu_score(generated_response: str, reference_response: str):
    """
    Calculates the BLEU score between the generated response and the reference response.

    Args:
        generated_response (str): The generated response.
        reference_response (str): The reference response.

    Returns:
        float: BLEU score.
    """
    reference_tokens = [reference_response.split()]
    generated_tokens = generated_response.split()
    return sentence_bleu(reference_tokens, generated_tokens)


