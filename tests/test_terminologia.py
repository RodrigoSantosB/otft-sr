"""Terminologia técnico-científica (R1): paridade numérica e varredura de termos vetados."""

import json
import re
import unicodedata
from pathlib import Path

import pytest

from otft_sr.experimento import ALGORITMOS_RS, construir_config

SNAPSHOT = Path(__file__).parent / "fixtures" / "configuracoes_busca_snapshot.json"


@pytest.mark.parametrize("algoritmo", ["pysr", "deap", "gplearn"])
@pytest.mark.parametrize("configuracao_busca", ["verificacao", "calibrada"])
def test_paridade_configuracoes_busca(algoritmo, configuracao_busca):
    """A renomeação não altera nenhum valor: config nova == snapshot gerado antes."""
    assert algoritmo in ALGORITMOS_RS
    esperado = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[f"{algoritmo}/{configuracao_busca}"]
    atual = construir_config(algoritmo, configuracao_busca, semente=0).model_dump(mode="json")
    assert atual == esperado


# ----------------------------------------------------------------------------- varredura
# Termos vetados (R1): terminologia informal substituída pelo glossário aprovado.
# A busca ignora caixa e acentos e usa limites de palavra.

RAIZ = Path(__file__).resolve().parents[1]

TERMOS_VETADOS = (
    r"or[cç]amentos?",
    r"motor(?:es)?",
    r"fuma[cç]a",
    r"smoke",
    r"budget",
    r"engine",
    r"carimbo",
    r"encadeamento",
    r"personalizad[oa]s?",
    r"bra[cç]os?",
    r"c[eé]lulas?",
)

# Exceções EXPLÍCITAS e mínimas (trechos removidos do texto antes da busca):
#  - ``celulas`` / "células)": célula de notebook Jupyter (termo técnico do nbformat) no
#    builder e no .ipynb; célula no sentido de ensaio experimental continua vetada.
#  - As chaves da API externa physics_llm_sr (``"mode": "pilot"``, ``baseline=``, ``sr.*``)
#    não contêm termos vetados e por isso não precisam de exceção.
EXCECOES = ("celulas", "células)")

# O próprio arquivo define os termos e por isso não é varrido.
ARQUIVO_DA_LISTA = Path(__file__).resolve()

ALVOS = [
    *sorted((RAIZ / "src").rglob("*.py")),
    *sorted(p for p in (RAIZ / "tests").rglob("*.py") if p.resolve() != ARQUIVO_DA_LISTA),
    RAIZ / "notebooks" / "build_01_rs_otft.py",
    RAIZ / "notebooks" / "01_rs_otft.ipynb",
    RAIZ / "notebooks" / "build_02_multi_llm_colab.py",
    RAIZ / "notebooks" / "02_multi_llm_descoberta_otft_colab.ipynb",
    *sorted((RAIZ / "scripts").glob("*.py")),
    RAIZ / "requirements-colab.txt",
    RAIZ / "README.md",
    *sorted((RAIZ / "references").glob("*.md")),
    *sorted(p for p in (RAIZ / "prompts").rglob("*") if p.is_file()),
]

# Alvos ainda não migrados: removidos quando a onda correspondente migrar o arquivo.
PENDENTES = {
}


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFD", texto)
    return "".join(c for c in sem_acento if unicodedata.category(c) != "Mn").lower()


def _ocorrencias(texto: str) -> list[str]:
    for exc in EXCECOES:
        texto = texto.replace(exc, "")
    padrao = re.compile(r"\b(?:" + "|".join(_normalizar(t) for t in TERMOS_VETADOS) + r")\b")
    return sorted(set(padrao.findall(_normalizar(texto))))


def _param(caminho: Path):
    rel = caminho.relative_to(RAIZ).as_posix()
    marcas = []
    if rel in PENDENTES:
        marcas.append(
            pytest.mark.xfail(strict=True, reason=f"migrado na onda {PENDENTES[rel]}")
        )
    return pytest.param(caminho, id=rel, marks=marcas)


@pytest.mark.parametrize("caminho", [_param(p) for p in ALVOS])
def test_sem_termos_vetados(caminho):
    achados = _ocorrencias(caminho.read_text(encoding="utf-8"))
    assert not achados, f"termos vetados em {caminho.name}: {achados}"


def test_varredura_detecta_termo_vetado():
    """Sanidade do detector: acento, caixa e exceções."""
    assert _ocorrencias("ORÇAMENTO e Motores") == ["motores", "orcamento"]
    assert _ocorrencias("celulas.append e 3 células)") == []
    assert _ocorrencias("célula do ensaio") == ["celula"]
