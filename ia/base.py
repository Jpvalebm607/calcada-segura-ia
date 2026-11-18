"""Base de conhecimento: baixa as normas oficiais e quebra cada uma por artigo.

Cada trecho guarda a fonte, o artigo e a URL, para a resposta poder citar
de onde veio. Textos de leis e decretos não têm direito autoral no Brasil
(Lei 9.610/1998, art. 8º, IV), então podem compor a base livremente.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import requests

LICENCA = "Texto oficial de norma jurídica, sem direito autoral (Lei 9.610/1998, art. 8º, IV)"

FONTES = [
    {
        "id": "D59671",
        "nome": "Decreto Municipal nº 59.671/2020 (padronização das calçadas de SP)",
        "url": "https://saopaulo.sp.leg.br/iah/fulltext/decretos/D59671.pdf",
        "formato": "pdf",
    },
    {
        "id": "L15442",
        "nome": "Lei Municipal nº 15.442/2011 (manutenção de passeios em SP)",
        "url": "https://www.saopaulo.sp.leg.br/iah/fulltext/leis/L15442.pdf",
        "formato": "pdf",
    },
    {
        "id": "LBI",
        "nome": "Lei nº 13.146/2015 (Lei Brasileira de Inclusão)",
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/l13146.htm",
        "formato": "html",
    },
    {
        "id": "L10098",
        "nome": "Lei nº 10.098/2000 (acessibilidade)",
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l10098.htm",
        "formato": "html",
    },
]

TAMANHO_MAX = 1400  # caracteres por trecho; artigos maiores viram partes


def baixar(fonte: dict, pasta: Path) -> Path:
    """Baixa o arquivo oficial. Se já existir em base/fontes/, reaproveita."""
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{fonte['id']}.{fonte['formato']}"
    if destino.exists() and destino.stat().st_size > 1000:
        return destino
    r = requests.get(fonte["url"], timeout=90, headers={"User-Agent": "Mozilla/5.0 (GS FIAP Calcada Segura)"})
    r.raise_for_status()
    destino.write_bytes(r.content)
    return destino


def _texto_html(caminho: Path) -> str:
    from bs4 import BeautifulSoup

    bruto = caminho.read_bytes()
    try:
        html = bruto.decode("utf-8")
    except UnicodeDecodeError:
        html = bruto.decode("cp1252", errors="replace")
    sopa = BeautifulSoup(html, "html.parser")
    # O Planalto mantém texto revogado riscado (<strike>): fica de fora da base
    for tag in sopa.find_all(["strike", "s", "del", "script", "style"]):
        tag.decompose()
    return sopa.get_text("\n")


def _texto_pdf(caminho: Path) -> str:
    from pypdf import PdfReader

    return "\n".join((p.extract_text() or "") for p in PdfReader(str(caminho)).pages)


def normalizar(texto: str) -> str:
    texto = texto.replace("\xa0", " ").replace("​", "")
    texto = re.sub(r"-\n(?=[a-zà-ú])", "", texto)  # palavra quebrada no fim da linha do PDF
    return re.sub(r"\s+", " ", texto).strip()


def extrair_texto(caminho: Path, formato: str) -> str:
    bruto = _texto_html(caminho) if formato == "html" else _texto_pdf(caminho)
    return normalizar(bruto)


# "Art. 1º", "Art. 10.", "Art. 4o" no início de um dispositivo (com A maiúsculo:
# as remissões no meio do texto costumam ser "art." minúsculo)
PADRAO_ARTIGO = re.compile(r"(?=(?<![\w])Art\.\s?\d+)")


def _rotulo_artigo(trecho: str) -> str:
    m = re.match(r"Art\.\s?(\d+)", trecho)
    return f"Art. {m.group(1)}" if m else "Preâmbulo"


def _partes(texto: str) -> list[str]:
    if len(texto) <= TAMANHO_MAX:
        return [texto]
    frases = re.split(r"(?<=[.;:])\s+(?=(?:§|[IVXLC]+ ?[-–]|[a-z]\)|[A-ZÀ-Ú]))", texto)
    partes, atual = [], ""
    for f in frases:
        if atual and len(atual) + len(f) + 1 > TAMANHO_MAX:
            partes.append(atual)
            atual = f
        else:
            atual = f"{atual} {f}".strip()
    if atual:
        partes.append(atual)
    return partes


def quebrar_por_artigo(texto: str, fonte: dict) -> list[dict]:
    blocos = [b.strip() for b in PADRAO_ARTIGO.split(texto) if b.strip()]
    trechos, vistos = [], {}
    for bloco in blocos:
        artigo = _rotulo_artigo(bloco)
        for i, parte in enumerate(_partes(bloco), start=1):
            if i > 1:  # parte continuada leva o número do artigo para não perder o contexto
                parte = f"{artigo} (continuação): {parte}"
            base_id = f"{fonte['id']}-{artigo.replace('Art. ', 'art')}".replace("Preâmbulo", "preambulo")
            vistos[base_id] = vistos.get(base_id, 0) + 1
            n = vistos[base_id]
            trechos.append({
                "id": f"{base_id}-{n}",
                "fonte": fonte["id"],
                "documento": fonte["nome"],
                "artigo": artigo,
                "texto": parte,
                "url": fonte["url"],
            })
    return [t for t in trechos if len(t["texto"]) >= 40]


def montar_base(pasta_base: Path) -> list[dict]:
    """Baixa todas as fontes, gera base/trechos.jsonl e base/tabela_base.csv."""
    pasta_base = Path(pasta_base)
    todos, linhas = [], []
    for fonte in FONTES:
        arquivo = baixar(fonte, pasta_base / "fontes")
        trechos = quebrar_por_artigo(extrair_texto(arquivo, fonte["formato"]), fonte)
        todos.extend(trechos)
        linhas.append({"fonte": fonte["id"], "documento": fonte["nome"], "url": fonte["url"],
                       "formato": fonte["formato"].upper(), "licenca": LICENCA, "trechos": len(trechos)})
    with open(pasta_base / "trechos.jsonl", "w", encoding="utf-8") as f:
        for t in todos:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    import pandas as pd

    pd.DataFrame(linhas).to_csv(pasta_base / "tabela_base.csv", index=False)
    return todos


def carregar_base(pasta_base: Path) -> list[dict]:
    with open(Path(pasta_base) / "trechos.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))
