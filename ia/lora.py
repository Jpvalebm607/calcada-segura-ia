"""Especialização por LoRA: ensina o modelo a responder a partir dos trechos,
citar o id do trecho e recusar quando a pergunta está fora da base.

Dados de treino (sintéticos, sem usar as perguntas de teste):
- positivos: o próprio modelo-base lê um trecho e escreve uma pergunta de cidadão
  e a resposta curta. Só fica o par cuja resposta está apoiada no trecho.
- negativos: perguntas fora do tema (outras cidades, outros serviços) com trechos
  quaisquer da base; o alvo é a recusa "Não consta na base".
"""
from __future__ import annotations

import json
import random
import re

from .avaliacao import norm
from .gerador import RECUSA, extrair_json, montar_mensagens

PEDIDO_PERGUNTA = (
    "Leia o trecho de norma abaixo e escreva UMA pergunta curta que um cidadão de São Paulo faria "
    "e cuja resposta esteja no trecho, e a resposta curta (uma frase). Responda só com JSON: "
    '{{"pergunta": "...", "resposta": "..."}}\n\nTrecho ({documento}, {artigo}):\n{texto}'
)

PERGUNTAS_FORA = [
    "Qual o horário de funcionamento do Poupatempo da Sé?",
    "Quanto custa a passagem de ônibus em São Paulo?",
    "Qual o prazo para pagar o IPVA deste ano?",
    "Como tirar a segunda via do RG?",
    "Qual a multa por excesso de velocidade em rodovia?",
    "Quais documentos preciso para abrir uma empresa?",
    "Qual a regra para poda de árvore em Belo Horizonte?",
    "Qual o salário de um agente vistor da Prefeitura?",
    "Como funciona a coleta seletiva no meu bairro?",
    "Qual o valor do IPTU de um apartamento em Moema?",
    "Quantas linhas tem o metrô de São Paulo?",
    "Qual a altura máxima de um muro em Campinas?",
    "Qual a previsão do tempo para amanhã?",
    "Como pedir a castração gratuita do meu cachorro?",
    "Qual a nota de corte do SISU para Medicina?",
    "Como trocar a titularidade da conta de luz?",
    "Qual o preço do aluguel de uma caçamba de entulho?",
    "Qual o limite de barulho permitido em festas no Rio de Janeiro?",
]


def _apoiada(resposta: str, texto: str) -> bool:
    """Pelo menos metade das palavras relevantes da resposta aparece no trecho."""
    palavras = [p for p in re.findall(r"\w+", norm(resposta)) if len(p) > 3]
    if not palavras:
        return False
    alvo = norm(texto)
    return sum(p in alvo for p in palavras) / len(palavras) >= 0.5


def gerar_pares(gerador, trechos: list[dict], n: int = 200, seed: int = 42) -> list[dict]:
    rnd = random.Random(seed)
    candidatos = [t for t in trechos if t["artigo"] != "Preâmbulo" and len(t["texto"]) > 120]
    pares = []
    for t in rnd.sample(candidatos, min(n, len(candidatos))):
        msg = [{"role": "user", "content": PEDIDO_PERGUNTA.format(**t)}]
        bruto = gerador.gerar(msg, max_new_tokens=120)
        m = re.search(r"\{.*\}", bruto, re.S)
        try:
            dado = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            continue
        p, r = str(dado.get("pergunta", "")).strip(), str(dado.get("resposta", "")).strip()
        if len(p) > 10 and len(r) > 3 and _apoiada(r, t["texto"]):
            pares.append({"pergunta": p, "resposta": r, "trecho": t})
    return pares


def montar_exemplos(pares: list[dict], trechos: list[dict], buscador, n_neg_por_pergunta: int = 3,
                    seed: int = 42) -> list[dict]:
    rnd = random.Random(seed)
    exemplos = []
    for par in pares:
        t = par["trecho"]
        contexto = [x for _, x in buscador.buscar(par["pergunta"], 3)]
        if t["id"] not in {x["id"] for x in contexto}:
            contexto = [t] + contexto[:2]
        rnd.shuffle(contexto)
        alvo = {"resposta": par["resposta"], "fonte": f"{t['documento']}, {t['artigo']}",
                "trecho_id": t["id"], "nao_consta": False}
        exemplos.append({"mensagens": montar_mensagens(par["pergunta"], contexto),
                         "alvo": json.dumps(alvo, ensure_ascii=False)})
    for pergunta in PERGUNTAS_FORA:
        for _ in range(n_neg_por_pergunta):
            contexto = rnd.sample(trechos, 3)
            exemplos.append({"mensagens": montar_mensagens(pergunta, contexto),
                             "alvo": json.dumps(RECUSA, ensure_ascii=False)})
    rnd.shuffle(exemplos)
    return exemplos


def treinar(gerador, exemplos: list[dict], pasta_saida: str, epocas: int = 2, lr: float = 2e-4,
            max_len: int = 1024) -> list[dict]:
    """Treina o adaptador LoRA em cima do modelo do gerador e o deixa ativo no gerador."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    tok, modelo = gerador.tok, gerador.model
    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    modelo.print_trainable_parameters()

    dados = []
    for ex in exemplos:
        prompt = tok.apply_chat_template(ex["mensagens"], tokenize=False, add_generation_prompt=True)
        ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
        ids_alvo = tok(ex["alvo"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (ids_prompt + ids_alvo)[-max_len:]
        rotulos = ([-100] * len(ids_prompt) + ids_alvo)[-max_len:]  # perda só na resposta
        dados.append({"input_ids": ids, "labels": rotulos})

    def juntar(lote):
        n = max(len(x["input_ids"]) for x in lote)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in lote]),
            "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in lote]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in lote]),
        }

    args = TrainingArguments(
        output_dir=pasta_saida, num_train_epochs=epocas, learning_rate=lr,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_ratio=0.05,
        lr_scheduler_type="cosine", logging_steps=10, save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available(), remove_unused_columns=False, seed=42,
    )
    treino = Trainer(model=modelo, args=args, train_dataset=dados, data_collator=juntar)
    treino.train()
    modelo.save_pretrained(pasta_saida)
    tok.save_pretrained(pasta_saida)
    try:  # volta ao modo de inferência normal
        modelo.base_model.model.gradient_checkpointing_disable()
    except AttributeError:
        pass
    modelo.config.use_cache = True
    modelo.eval()
    gerador.model = modelo
    return treino.state.log_history
