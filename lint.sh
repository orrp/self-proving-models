echo ">>> isort"
isort spm
echo ">>> black"
black spm
echo ">>> flake8"
flake8 spm
echo ">>> mypy"
mypy --cache-dir ../.mypy_cache spm
