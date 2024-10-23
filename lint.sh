echo ">>> isort"
isort spm
echo ">>> black"
black spm
echo ">>> flake8"
flake8 --toml-config pyproject.toml spm # explicitly pass toml-config so that this fails if flake8-pyproject not found
echo ">>> mypy"
mypy --cache-dir ../.mypy_cache spm
