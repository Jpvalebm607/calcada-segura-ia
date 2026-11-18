"""Geração fundamentada: LLM local responde em JSON, cita a fonte e recusa quando não consta."""
from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, ValidationError

MODELO_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

FORMATO = (
    'Responda SOMENTE com um objeto JSON, sem texto fora dele, no formato: '
    '{"resposta": "<resposta curta em português>", "fonte": "<norma e artigo>", '
    '"trecho_id": "<id do trecho usado>", "nao_consta": false}'
)

SISTEMA_RAG = (
    "Você é o assistente do Calçada Segura, que explica as regras de calçadas de São Paulo "
    "e de acessibilidade para cidadãos. Use APENAS os trechos fornecidos. "
    "Cite a norma, o artigo e o id do trecho usado. Se os trechos não trouxerem a resposta, "
    'devolva {"resposta": "Não consta na base.", "fonte": null, "trecho_id": null, "nao_consta": true}. '
    + FORMATO
)

SISTEMA_SEM_CONTEXTO = (
    "Você é o assistente do Calçada Segura, que explica as regras de calçadas de São Paulo "
    "e de acessibilidade para cidadãos. Responda com o que você sabe, citando a norma "
    "(nome e número) e o artigo. Se não souber, "
    'devolva {"resposta": "Não consta na base.", "fonte": null, "trecho_id": null, "nao_consta": true}. '
    + FORMATO
)

RECUSA = {"resposta": "Não consta na base.", "fonte": None, "trecho_id": None, "nao_consta": True}


class Resposta(BaseModel):
    resposta: str
    fonte: str | None = None
    trecho_id: str | None = None
    nao_consta: bool


def montar_mensagens(pergunta: str, trechos: list[dict] | None) -> list[dict]:
    if trechos is None:
        return [{"role": "system", "content": SISTEMA_SEM_CONTEXTO},
                {"role": "user", "content": f"Pergunta: {pergunta}"}]
    contexto = "\n\n".join(f"[{t['id']}] {t['documento']}, {t['artigo']}:\n{t['texto']}" for t in trechos)
    return [{"role": "system", "content": SISTEMA_RAG},
            {"role": "user", "content": f"Trechos:\n{contexto}\n\nPergunta: {pergunta}"}]


def extrair_json(texto: str) -> tuple[dict | None, bool]:
    """Pega o primeiro objeto JSON da saída e valida o esquema. Retorna (dados, json_valido)."""
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        return None, False
    try:
        return Resposta.model_validate(json.loads(m.group(0))).model_dump(), True
    except (json.JSONDecodeError, ValidationError):
        return None, False


class Gerador:
    def __init__(self, modelo: str = MODELO_LLM, adaptador: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(modelo)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=dtype, device_map="auto")
        if adaptador:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adaptador)
        self.model.eval()

    def gerar(self, mensagens: list[dict], max_new_tokens: int = 220) -> str:
        prompt = self.tok.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
        entrada = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            saida = self.model.generate(**entrada, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(saida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True)


def responder(pergunta: str, config: str, gerador: "Gerador", buscador=None, limiar: float | None = None,
              k: int = 3) -> dict:
    """Roda uma configuração: 'A_sem_contexto', 'B_rag_simples' ou 'C_rag_lora'."""
    t0 = time.time()
    recuperados, trechos = [], None
    if config != "A_sem_contexto":
        resultado = buscador.buscar(pergunta, k)
        recuperados = [(round(s, 4), t["id"]) for s, t in resultado]
        if limiar is not None and resultado[0][0] < limiar:
            # Recusa antes de chamar o modelo: nada na base é parecido o bastante
            return {"saida": dict(RECUSA), "json_valido": True, "recuperados": recuperados,
                    "bruto": "(recusa pelo limiar de similaridade)", "tempo_s": round(time.time() - t0, 2)}
        trechos = [t for _, t in resultado]
    bruto = gerador.gerar(montar_mensagens(pergunta, trechos))
    saida, valido = extrair_json(bruto)
    return {"saida": saida, "json_valido": valido, "recuperados": recuperados,
            "bruto": bruto, "tempo_s": round(time.time() - t0, 2)}
