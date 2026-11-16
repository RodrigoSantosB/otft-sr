"""Contexto físico de G1/G2 e do prompt de dispositivo (M1 C): princípios qualitativos
declarados em H-OTFT-21, sem forma matemática explícita nem constantes dos níveis."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from physics_llm_sr.symbolic.expression import parse_infix

from otft_sr.contexto import VARIANTE_PROMPT, carregar_biblioteca, contexto_fisico
from otft_sr.dados import construir_datasets
from otft_sr.formulacao import nivel_formulado
from otft_sr.gerador import montar_prompt
from otft_sr.niveis import FEATURES, escada

RAIZ = Path(__file__).resolve().parents[1]
PREMISSAS = RAIZ / "references" / "premissas_cientificas.md"

#: Frases de princípio qualitativo PERMITIDAS (conhecimento geral de OTFT, proposta §5.2).
#: Elas trazem informação estrutural sobre os regimes (N3: mobilidade crescente com o
#: excesso de tensão de porta; N4: subida exponencial até o piso de fuga) e estão
#: declaradas em H-OTFT-21. Alterar o texto exige atualizar esta lista e a premissa.
PRINCIPIOS_PERMITIDOS: dict[str, tuple[str, ...]] = {
    "prompts/gerador/contexto_fisico.txt": (
        "rises exponentially with V_GS",
        "masked by a constant leakage floor I_leak",
        "the charge grows with the gate overdrive V_GS - V_Th",
        "where I_D grows with V_DS and vanishes at V_DS = 0",
        "where I_D becomes nearly independent of V_DS",
        "the effective mobility is not constant and usually increases with the gate overdrive "
        "V_GS - V_Th",
    ),
    "prompts/dispositivo/base/system.txt": (
        "the effective mobility usually increases with the gate overdrive |V_GS - V_Th|",
        "subthreshold (below V_Th, current rises exponentially with V_GS)",
        "linear (V_DS small compared with V_GS - V_Th, current grows with V_DS)",
        "saturation (V_DS above V_GS - V_Th, current becomes nearly independent of V_DS)",
    ),
}

#: Formas matemáticas explícitas proibidas no texto de contexto físico (``cm^2`` em
#: unidades é permitido).
FORMAS_EXPLICITAS: tuple[str, ...] = (
    r"\*\*", r"(?<!cm)\^", r"\bexp\s*\(", r"\blog(?:10)?\s*\(", r"\bsqrt\s*\(",
    r"\(\s*V_GS\s*-\s*V_(?:Th|th0|T|x)\b",
)
NIVEIS = escada()


def _algarismos_significativos(v: float) -> int:
    return len(f"{abs(v):.15g}".split("e")[0].replace(".", "").strip("0"))


def _formatos(v: float) -> set[str]:
    return {repr(v), f"{v:g}", f"{v:.4g}", f"{v:.3g}"}


def _constantes_proibidas(n) -> set[str]:
    """Constantes do nível com >= 3 algarismos significativos ou em notação científica (as
    curtas, como I_leak = 65 pA, coincidem com valores dos próprios dados de treino)."""
    return {f for nome, v in n.constantes.items() if not nome.startswith("S_teorico")
            and (_algarismos_significativos(float(v)) >= 3 or "e" in repr(float(v)))
            for f in _formatos(float(v))}


def _contexto_gerador(n) -> str:
    """Linhas que G1 acrescenta a G0 (o contexto físico efetivamente enviado)."""
    ds = construir_datasets(n, 0.0, 0, 10, 10)
    _, u0 = montar_prompt(n, "G0", ds["treino"], 1, 3)
    _, u1 = montar_prompt(n, "G1", ds["treino"], 1, 3)
    base = set(u0.splitlines())
    return "\n".join(x for x in u1.splitlines() if x not in base)


def _fundo_dispositivo() -> str:
    texto = (RAIZ / "prompts/dispositivo/base/system.txt").read_text(encoding="utf-8")
    return texto.split("### DEVICE BACKGROUND")[1].split("### METRICS")[0]


def _prompts_completos(n) -> dict[str, str]:
    ds = construir_datasets(n, 0.0, 0, 10, 10)
    out = {}
    for g in ("G0", "G1", "G2"):
        s, u = montar_prompt(n, g, ds["treino"], 1, 3)
        out[g] = s + "\n" + u
    for f in ("linear", "log10"):
        for c in ("B", "C"):
            r = carregar_biblioteca(c).render(parse_infix("V_GS * V_DS", FEATURES),
                                              contexto_fisico(nivel_formulado(n, f)),
                                              VARIANTE_PROMPT)
            out[f"M1 {c} {f}"] = r.system + "\n" + r.user
    return out


@pytest.mark.parametrize("arquivo", sorted(PRINCIPIOS_PERMITIDOS))
def test_principios_qualitativos_permitidos_estao_no_prompt(arquivo):
    texto = " ".join((RAIZ / arquivo).read_text(encoding="utf-8").split())
    for frase in PRINCIPIOS_PERMITIDOS[arquivo]:
        assert frase in texto, frase


def test_premissa_h_otft_21_declara_o_contexto_qualitativo():
    texto = " ".join(PREMISSAS.read_text(encoding="utf-8").split())
    m = re.search(r"\*\*H-OTFT-21 — (.*?)(?=- \*\*H-OTFT-|\Z)", texto)
    assert m, "H-OTFT-21 ausente"
    premissa = m.group(1)
    for trecho in ("princípios qualitativos", "G0 × G1", "B × C", "§5.2",
                   "prompts/gerador/contexto_fisico.txt", "prompts/dispositivo/base/system.txt",
                   "tests/test_contexto_fisico_prompts.py"):
        assert trecho in premissa, trecho


@pytest.mark.parametrize("n", NIVEIS, ids=lambda n: n.id)
def test_contexto_do_gerador_sem_forma_matematica_explicita(n):
    contexto = _contexto_gerador(n)
    assert "Physical context" in contexto
    for padrao in FORMAS_EXPLICITAS:
        assert not re.search(padrao, contexto), padrao


def test_fundo_do_dispositivo_sem_forma_matematica_explicita():
    fundo = _fundo_dispositivo()
    assert "hopping" in fundo
    for padrao in FORMAS_EXPLICITAS:
        assert not re.search(padrao, fundo), padrao


@pytest.mark.parametrize("n", NIVEIS, ids=lambda n: n.id)
def test_prompts_sem_constantes_nem_fator_de_excesso_de_tensao(n):
    proibidas = _constantes_proibidas(n)
    assert proibidas
    for nome, texto in _prompts_completos(n).items():
        assert not re.search(FORMAS_EXPLICITAS[-1], texto), nome
        for c in proibidas:
            assert not re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", texto), (nome, c)


def test_detector_reprova_forma_explicita():
    """O detector não é vácuo: frases com forma matemática explícita são reprovadas."""
    for frase in ("I_D = K * (V_GS - V_Th) ** 2", "mu ~ (V_GS - V_Th)^gamma",
                  "I_D ~ exp(V_GS / S)", "the overdrive (V_GS - V_Th) enters squared"):
        assert any(re.search(p, frase) for p in FORMAS_EXPLICITAS), frase
    n3 = next(n for n in NIVEIS if n.id == "N3")
    assert {"3.18", "8.57", "2.2e-05"} <= _constantes_proibidas(n3)
    n4 = next(n for n in NIVEIS if n.id == "N4")
    assert "65.0" not in _constantes_proibidas(n4)
