#!/usr/bin/env python3
"""
Lê VCIP-Transfusion.ipynb, extrai os imports das células de código
e regenera requirements.txt com as versões instaladas.

Uso:
    python3 .claude/update_requirements.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

NOTEBOOK     = Path(__file__).parent.parent / 'VCIP-Transfusion.ipynb'
REQUIREMENTS = Path(__file__).parent.parent / 'requirements.txt'

# ── Map: nome do módulo Python → nome do pacote pip ──────────────────────────
MODULE_TO_PKG = {
    'pandas':            'pandas',
    'numpy':             'numpy',
    'matplotlib':        'matplotlib',
    'seaborn':           'seaborn',
    'sklearn':           'scikit-learn',
    'scipy':             'scipy',
    'pyarrow':           'pyarrow',
    'torch':             'torch',
    'torchvision':       'torchvision',
    'pytorch_lightning': 'pytorch-lightning',
    'hydra':             'hydra-core',
    'omegaconf':         'omegaconf',
    'geomloss':          'geomloss',
    'plotly':            'plotly',
    'IPython':           'ipython',
    'ipywidgets':        'ipywidgets',
    'statsmodels':       'statsmodels',
    'xgboost':           'xgboost',
    'lightgbm':          'lightgbm',
    'shap':              'shap',
    'PIL':               'Pillow',
    'cv2':               'opencv-python',
    'requests':          'requests',
    'tqdm':              'tqdm',
    'yaml':              'PyYAML',
}

# ── Módulos da stdlib (ignorar) ───────────────────────────────────────────────
STDLIB = {
    'os', 'sys', 're', 'json', 'pathlib', 'datetime', 'collections',
    'itertools', 'functools', 'typing', 'math', 'random', 'copy',
    'warnings', 'logging', 'time', 'io', 'abc', 'enum', 'dataclasses',
    'contextlib', 'operator', 'string', 'struct', 'hashlib', 'subprocess',
    'threading', 'multiprocessing', 'gc', 'inspect', 'traceback', 'glob',
    'shutil', 'tempfile', 'pickle', 'csv', 'textwrap', 'pprint',
}

# ── Pacotes sempre incluídos (runtime do notebook + parquet) ──────────────────
ALWAYS_INCLUDE = ['jupyter', 'notebook', 'ipykernel', 'pyarrow']

# ── Comentários de seção (pkg → seção) ───────────────────────────────────────
SECTIONS = {
    'jupyter':           'Notebook runtime',
    'notebook':          'Notebook runtime',
    'ipykernel':         'Notebook runtime',
    'pandas':            'Data',
    'numpy':             'Data',
    'pyarrow':           'Data',
    'matplotlib':        'Visualização',
    'seaborn':           'Visualização',
    'plotly':            'Visualização',
    'scikit-learn':      'ML / Stats',
    'scipy':             'ML / Stats',
    'statsmodels':       'ML / Stats',
    'xgboost':           'ML / Stats',
    'lightgbm':          'ML / Stats',
    'shap':              'ML / Stats',
    'torch':             'Deep Learning',
    'torchvision':       'Deep Learning',
    'pytorch-lightning': 'Deep Learning',
    'hydra-core':        'Config',
    'omegaconf':         'Config',
}


def get_version(pkg: str) -> str | None:
    """Retorna a versão instalada de um pacote pip."""
    try:
        result = subprocess.run(
            [sys.executable, '-c',
             f'import importlib.metadata; print(importlib.metadata.version("{pkg}"))'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def extract_imports(notebook_path: Path) -> set[str]:
    """Extrai todos os top-level imports das células de código do notebook."""
    nb = json.loads(notebook_path.read_text())
    top_modules: set[str] = set()

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        # Captura: import X, import X.y, from X import ..., from X.y import ...
        for m in re.finditer(
            r'^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)',
            source, re.MULTILINE
        ):
            top_modules.add(m.group(1))

    return top_modules


def build_requirements(modules: set[str]) -> dict[str, str | None]:
    """Mapeia módulos para pacotes pip com suas versões instaladas."""
    pkgs: dict[str, str | None] = {}

    # Imports do notebook
    for mod in sorted(modules):
        if mod in STDLIB:
            continue
        pkg = MODULE_TO_PKG.get(mod, mod)
        pkgs[pkg] = get_version(pkg)

    # Sempre inclui runtime do notebook
    for pkg in ALWAYS_INCLUDE:
        if pkg not in pkgs:
            pkgs[pkg] = get_version(pkg)

    return pkgs


def write_requirements(pkgs: dict[str, str | None], out_path: Path) -> None:
    """Escreve requirements.txt agrupado por seção."""
    # Agrupa por seção
    sections: dict[str, list[tuple[str, str | None]]] = {}
    for pkg, version in sorted(pkgs.items()):
        sec = SECTIONS.get(pkg, 'Outros')
        sections.setdefault(sec, []).append((pkg, version))

    section_order = [
        'Notebook runtime', 'Data', 'Visualização',
        'ML / Stats', 'Deep Learning', 'Config', 'Outros'
    ]

    lines = [
        '# Auto-generated from VCIP-Transfusion.ipynb imports\n',
        '# Run: pip install -r requirements.txt\n',
        '\n',
    ]

    for sec in section_order:
        if sec not in sections:
            continue
        lines.append(f'# ── {sec} {"─" * max(1, 68 - len(sec))}──\n')
        for pkg, version in sections[sec]:
            if version:
                lines.append(f'{pkg}>={version}\n')
            else:
                lines.append(f'{pkg}\n')
        lines.append('\n')

    out_path.write_text(''.join(lines).rstrip() + '\n')


def main() -> None:
    if not NOTEBOOK.exists():
        print(f'Notebook não encontrado: {NOTEBOOK}', file=sys.stderr)
        sys.exit(1)

    modules = extract_imports(NOTEBOOK)
    pkgs    = build_requirements(modules)

    write_requirements(pkgs, REQUIREMENTS)

    installed = sum(1 for v in pkgs.values() if v)
    missing   = sum(1 for v in pkgs.values() if v is None)
    print(f'requirements.txt atualizado: {installed} pacotes com versão'
          + (f', {missing} sem versão instalada' if missing else ''))


if __name__ == '__main__':
    main()
