from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from constants import DEVICE

class ZweigRewardFunction:
    def __init__(self, tokenizer, reference_texts, style_weight=0.7, format_weight=0.3):
        self.tokenizer = tokenizer
        self.style_weight = style_weight
        self.format_weight = format_weight
        
        # Precompute reference vectors during initialization
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            preprocessor=lambda text: self.clean_text(text)
        )
        self.ref_vecs = self.vectorizer.fit_transform(reference_texts)

    def clean_text(self, text):
        return text.replace(self.tokenizer.pad_token, "").strip()

    def compute_style_reward(self, texts):
        cleaned_texts = [self.clean_text(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned_texts)
        sim_scores = cosine_similarity(vecs, self.ref_vecs).mean(axis=1)
        return torch.tensor(sim_scores, dtype=torch.float32, requires_grad=False)  # Detached

    def compute_format_reward(self, texts):
        scores = []
        for text in texts:
            score = 0.0
            if "<stefan_zweig>" in text and "</stefan_zweig>" in text:
                score += 1.0
                if text.count("<stefan_zweig>") == text.count("</stefan_zweig>"):
                    score += 0.5
            if text.endswith("</stefan_zweig><|end_of_text|>"):
                score += 1.0
            scores.append(score / 2.5)
        return torch.tensor(scores, dtype=torch.float32, requires_grad=False)  # Detached

    def __call__(self, prompts, completions, **kwargs):        
        # Style reward
        cleaned = [self.clean_text(t) for t in completions]
        vecs = self.vectorizer.transform(cleaned)
        style_scores = cosine_similarity(vecs, self.ref_vecs).mean(axis=1)
        
        # Format reward
        format_scores = []
        for text in completions:
            score = 0
            if "<stefan_zweig>" in text and "</stefan_zweig>" in text:
                score += 1.5 if text.count("<stefan_zweig>") == text.count("</stefan_zweig>") else 1.0
            score += 1.0 if text.endswith("</stefan_zweig><|end_of_text|>") else 0
            format_scores.append(score / 2.5)
        
        # Combine scores
        total = torch.tensor(
            [0.7 * s + 0.3 * f for s, f in zip(style_scores, format_scores)],
            dtype=torch.float32,
            device=DEVICE
        )
        
        return total