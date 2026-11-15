"""Documentação: caminhos citados existem e premissas H-OTFT-01..20 estão íntegras."""

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
README = RAIZ / "README.md"
PREMISSAS = RAIZ / "references" / "premissas_cientificas.md"

PREFIXOS = ("src", "tests", "notebooks", "prompts", "data", "references")
# results/ é gerado pelos notebooks e ignorado pelo git: não é verificado.
RE_CAMINHO = re.compile(r"^(?:%s)/[^\s*<>{}]*$" % "|".join(PREFIXOS))


def _caminhos_citados(texto: str) -> set[str]:
    citados = set(re.findall(r"`([^`\n]+)`", texto))
    # bloco de estrutura: primeiro token de cada linha de um bloco cercado
    for bloco in re.findall(r"```[^\n]*\n(.*?)```", texto, flags=re.S):
        for linha in bloco.splitlines():
            if linha.strip():
                citados.add(linha.split()[0])
    return {c for c in citados if RE_CAMINHO.match(c)}


def test_caminhos_citados_no_readme_existem():
    caminhos = _caminhos_citados(README.read_text(encoding="utf-8"))
    assert caminhos, "nenhum caminho detectado: o detector está quebrado"
    ausentes = sorted(c for c in caminhos if not (RAIZ / c).exists())
    assert ausentes == []


def test_caminhos_citados_nas_premissas_existem():
    caminhos = _caminhos_citados(PREMISSAS.read_text(encoding="utf-8"))
    ausentes = sorted(c for c in caminhos if not (RAIZ / c).exists())
    assert ausentes == []


def test_premissas_h_otft_01_a_20_contiguas():
    ids = re.findall(r"^- \*\*H-OTFT-(\d{2}[a-z]?)", PREMISSAS.read_text(encoding="utf-8"), flags=re.M)
    numeros = [int(i[:2]) for i in ids]
    assert sorted(set(numeros)) == list(range(1, 21))
    assert numeros == sorted(numeros)  # em ordem
    assert len(ids) == len(set(ids))  # sem duplicatas
    extras = [i for i in ids if i[2:]]
    assert extras == ["07b"]  # única variante alfabética permitida
