.PHONY: get-data clean docs

get-data:
	@ mkdir -p data
	@ wget -P data/ "https://zenodo.org/record/3384388/files/0_dB_pump.zip"
	@ unzip data/0_dB_pump.zip -d data/

clean:
	@ find . -iname '__pycache__' -exec rm -rf {} \;        2>/dev/null || true
	@ find . -iname '*~' -exec rm -rf {} \;                 2>/dev/null || true
docs:
	@ sphinx-apidoc --ext-autodoc -o docs tools
	@ sphinx-build -d docs/_build/doctrees docs docs/_build/html
