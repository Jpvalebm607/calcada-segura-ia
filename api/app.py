"""API do assistente Calçada Segura.

    uvicorn api.app:app --port 8001           # modo completo (LLM + LoRA)
    MODO=leve uvicorn api.app:app --port 8001 # modo leve: só busca, sem LLM

POST /perguntar {"pergunta": "..."} -> JSON com resposta, fonte, trecho_id, nao_consta.

Resiliência: se o modelo não carregar (sem GPU/memória) ou MODO=leve, a API
continua respondendo com os trechos da norma em vez de cair ("degrada com elegância").
LGPD: a API não grava a pergunta nem dados de quem pergunta.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from ia.base import carregar_base  # noqa: E402
from ia.busca import Buscador  # noqa: E402
from ia.gerador import RECUSA, responder  # noqa: E402

trechos = carregar_base(RAIZ / "base")
buscador = Buscador.carregar(trechos, RAIZ / "base")
LIMIAR = json.loads((RAIZ / "avaliacao" / "limiar.json").read_text())["limiar"]

gerador = None
if os.getenv("MODO", "completo") != "leve":
    try:
        from ia.gerador import Gerador

        adaptador = RAIZ / "modelo" / "lora_calcada"
        gerador = Gerador(adaptador=str(adaptador) if adaptador.exists() else None)
    except Exception as erro:  # sem GPU/memória: segue em modo leve
        print("LLM indisponível, usando modo leve:", erro)

app = FastAPI(title="Assistente Calçada Segura", version="1.0")


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "modo": "completo" if gerador else "leve", "trechos": len(trechos)}


@app.post("/perguntar")
def perguntar(p: Pergunta):
    if gerador is None:
        achados = buscador.buscar(p.pergunta, 3)
        if achados[0][0] < LIMIAR:
            return {**RECUSA, "modo": "leve", "trechos": []}
        _, t = achados[0]
        return {
            "resposta": "Assistente em modo leve: veja o trecho da norma mais relacionado.",
            "fonte": f"{t['documento']}, {t['artigo']}", "trecho_id": t["id"], "nao_consta": False,
            "modo": "leve",
            "trechos": [{"id": x["id"], "score": round(s, 3), "texto": x["texto"][:400]} for s, x in achados],
        }
    r = responder(p.pergunta, "C_rag_lora", gerador, buscador, LIMIAR)
    saida = r["saida"] or {**RECUSA, "resposta": "Não consegui gerar uma resposta válida agora."}
    return {**saida, "modo": "completo", "trechos": [{"id": i, "score": s} for s, i in r["recuperados"]]}
