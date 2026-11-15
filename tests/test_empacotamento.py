"""Pacote do Colab: sem results/, sem .cache/, sem segredos e com VERSAO_PACOTE.json."""

from __future__ import annotations

import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("empacotar", RAIZ / "scripts" / "empacotar_para_colab.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def pacote(tmp_path_factory):
    return _script().empacotar(RAIZ, tmp_path_factory.mktemp("dist"))


def test_zip_contem_o_necessario(pacote):
    assert pacote.name.startswith("otft-sr-") and pacote.suffix == ".zip"
    nomes = set(zipfile.ZipFile(pacote).namelist())
    for esperado in ("src/otft_sr/campanha.py", "prompts/gerador/sistema.txt", "data/org1/inputs_Org1_SB.json",
                     "references/premissas_cientificas.md", "requirements-colab.txt", "README.md",
                     "notebooks/02_multi_llm_descoberta_otft_colab.ipynb", "VERSAO_PACOTE.json"):
        assert esperado in nomes, esperado
    versao = json.loads(zipfile.ZipFile(pacote).read("VERSAO_PACOTE.json"))
    assert versao["commit_curto"] in pacote.name


def test_zip_sem_resultados_cache_git_nem_segredos(pacote):
    from otft_sr.persistencia import _PADROES_SEGREDO

    z = zipfile.ZipFile(pacote)
    for nome in z.namelist():
        partes = Path(nome).parts
        assert not {"results", ".cache", ".git", "__pycache__", "dist"} & set(partes), nome
        assert not nome.endswith((".pyc", ".env", ".key", ".pem")), nome
        try:
            texto = z.read(nome).decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert not _PADROES_SEGREDO.search(texto), nome


def test_segredo_aborta_o_pacote(tmp_path):
    m = _script()
    for d in m.DIRETORIOS:
        (tmp_path / d).mkdir()
    shutil.copy(RAIZ / "README.md", tmp_path / "README.md")
    (tmp_path / "src" / "config.py").write_text('CHAVE = "sk-' + "a" * 30 + '"\n')
    (tmp_path / "src" / ".env").write_text("OPENAI_API_KEY=x\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "m.csv").write_text("1")
    assert Path("src/.env") not in m.arquivos_do_pacote(tmp_path)
    with pytest.raises(m.SegredoEncontrado):
        m.empacotar(tmp_path, tmp_path / "dist")
