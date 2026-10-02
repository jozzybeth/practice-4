import re
import nltk
import sacrebleu
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer


def get_model_slug(model_name: str) -> str:
    """Генерация короткого имени модели для файлов предиктов."""
    clean_name = model_name.split("/")[-1].lower()
    clean_name = re.sub(r"[^a-z0-9]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return clean_name


def setup_nltk():
    """Загрузка необходимых пакетов NLTK."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def first_last_sentences(dialogue: str) -> str:
    """baseline: первое и последнее предложения."""
    dialogue = re.sub(r"#Person\d+#:\s*", "", str(dialogue))
    sentences = sent_tokenize(dialogue)

    if len(sentences) == 0:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    return f"{sentences[0]} {sentences[-1]}"


def calculate_metrics(predictions: list, references: list) -> dict:
    """Расчёт метрик BLEU и ROUGE-1/2/L."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    rouge_scores = [
        scorer.score(ref, pred) for pred, ref in zip(predictions, references)
    ]

    r1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
    r2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
    rL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)

    return {
        "BLEU": bleu.score,
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rL,
    }