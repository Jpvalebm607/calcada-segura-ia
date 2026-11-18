"""Busca semântica nos trechos da base (embeddings E5 multilíngue + produto interno).

A base tem poucas centenas de trechos, então uma busca exata com numpy é
suficiente e mais simples de reproduzir que um índice FAISS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELO_EMB = "intfloat/multilingual-e5-base"


def _texto_passagem(t: dict) -> str:
    # O E5 pede os prefixos "passage:" e "query:"; o nome da norma ajuda a busca
    return f"passage: {t['documento']}, {t['artigo']}: {t['texto']}"


class Buscador:
    def __init__(self, trechos: list[dict], modelo: str = MODELO_EMB, embeddings: np.ndarray | None = None):
        from sentence_transformers import SentenceTransformer

        self.trechos = trechos
        self.modelo = SentenceTransformer(modelo)
        if embeddings is None:
            embeddings = self.modelo.encode([_texto_passagem(t) for t in trechos], batch_size=32,
                                            normalize_embeddings=True, show_progress_bar=True)
        self.emb = np.asarray(embeddings, dtype=np.float32)

    def buscar(self, pergunta: str, k: int = 3) -> list[tuple[float, dict]]:
        q = self.modelo.encode([f"query: {pergunta}"], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.trechos[i]) for i in idx]

    def salvar(self, pasta: Path) -> None:
        np.save(Path(pasta) / "embeddings.npy", self.emb)

    @classmethod
    def carregar(cls, trechos: list[dict], pasta: Path, modelo: str = MODELO_EMB) -> "Buscador":
        return cls(trechos, modelo, embeddings=np.load(Path(pasta) / "embeddings.npy"))
