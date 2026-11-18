"""Métricas: recall@k da busca, escolha do limiar e placar das respostas."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .base import sem_acento

APELIDOS = {
    "D59671": ["59.671", "59671"],
    "L15442": ["15.442", "15442"],
    "LBI": ["13.146", "13146", "inclusao"],
    "L10098": ["10.098", "10098"],
}


def norm(texto: str | None) -> str:
    return re.sub(r"\s+", " ", sem_acento(texto or "")).strip()


def carregar_casos(caminho: Path) -> list[dict]:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def relevantes(caso: dict, trechos: list[dict]) -> set[str]:
    """Trechos que respondem o caso: da fonte aceita e contendo um dos termos esperados."""
    termos = [norm(p) for p in caso["trecho_contem"]]
    return {t["id"] for t in trechos
            if t["fonte"] in caso["fontes_aceitas"] and any(p in norm(t["texto"]) for p in termos)}


def avaliar_busca(casos: list[dict], trechos: list[dict], buscador, k_max: int = 3) -> pd.DataFrame:
    linhas = []
    for c in casos:
        res = buscador.buscar(c["pergunta"], k_max)
        ids = [t["id"] for _, t in res]
        rel = relevantes(c, trechos) if c["tipo"] == "consta" else set()
        linhas.append({
            "id": c["id"], "tipo": c["tipo"], "score_top1": round(res[0][0], 4), "top3": ids,
            "n_relevantes": len(rel),
            "acerto@1": bool(rel & set(ids[:1])) if rel else None,
            "acerto@3": bool(rel & set(ids[:3])) if rel else None,
        })
    return pd.DataFrame(linhas)


def escolher_limiar(df_busca: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Limiar de similaridade que melhor separa perguntas que constam das que não constam.

    Critério: maior acurácia balanceada (média entre % de 'consta' acima do limiar
    e % de 'não consta' abaixo dele). Empate: o menor limiar, para recusar menos.
    """
    consta = df_busca[df_busca.tipo == "consta"].score_top1
    nao = df_busca[df_busca.tipo == "nao_consta"].score_top1
    candidatos = sorted(set(df_busca.score_top1))
    curva = []
    for th in candidatos:
        tpr = (consta >= th).mean()
        tnr = (nao < th).mean()
        curva.append({"limiar": th, "consta_aceitas": tpr, "nao_consta_recusadas": tnr, "balanceada": (tpr + tnr) / 2})
    curva = pd.DataFrame(curva)
    melhor = curva.sort_values(["balanceada", "limiar"], ascending=[False, True]).iloc[0]
    return float(melhor.limiar), curva


def cita_fonte(saida: dict, fontes: list[str]) -> bool:
    trecho = saida.get("trecho_id") or ""
    if any(trecho.startswith(f + "-") for f in fontes):
        return True
    texto_fonte = norm(saida.get("fonte"))
    return any(a in texto_fonte for f in fontes for a in APELIDOS.get(f, []))


def pontuar(caso: dict, resultado: dict) -> dict:
    saida, valido = resultado["saida"], resultado["json_valido"]
    r = {"json_valido": valido, "correto": False, "alucinacao": False,
         "recusa_correta": False, "recusa_indevida": False, "cita_fonte_certa": False}
    if not valido or saida is None:
        r["alucinacao"] = caso["tipo"] == "nao_consta"  # inventou formato em vez de recusar
        return r
    recusou = bool(saida.get("nao_consta"))
    if caso["tipo"] == "nao_consta":
        r["correto"] = r["recusa_correta"] = recusou
        r["alucinacao"] = not recusou
        return r
    if recusou:
        r["recusa_indevida"] = True
        return r
    conteudo_ok = any(norm(p) in norm(saida.get("resposta")) for p in caso["palavras_resposta"])
    r["cita_fonte_certa"] = cita_fonte(saida, caso["fontes_aceitas"])
    r["correto"] = conteudo_ok and r["cita_fonte_certa"]
    r["alucinacao"] = not r["correto"]  # respondeu algo errado ou sem a fonte certa
    return r


def placar(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config")
    tab = pd.DataFrame({
        "acuracia_%": g.correto.mean() * 100,
        "alucinacao_%": g.alucinacao.mean() * 100,
        "json_valido_%": g.json_valido.mean() * 100,
        "recusa_correta_%": df[df.tipo == "nao_consta"].groupby("config").recusa_correta.mean() * 100,
        "recusa_indevida_%": df[df.tipo == "consta"].groupby("config").recusa_indevida.mean() * 100,
        "cita_fonte_certa_%": df[df.tipo == "consta"].groupby("config").cita_fonte_certa.mean() * 100,
        "tempo_medio_s": g.tempo_s.mean(),
    })
    return tab.round(1)
