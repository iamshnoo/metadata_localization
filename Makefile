.PHONY: help install-culture-map culture-map-help qa-help pretraining-recipes clean-pycache

help:
	@printf '%s\n' 'Common repository commands:'
	@printf '%s\n' '  make install-culture-map  Install the reusable culture-map package in editable mode'
	@printf '%s\n' '  make culture-map-help     Show the culture-map CLI help'
	@printf '%s\n' '  make qa-help              Show the QA dataset builder help'
	@printf '%s\n' '  make pretraining-recipes  List Nanotron pretraining recipe manifests'
	@printf '%s\n' '  make clean-pycache        Remove local Python cache directories'

install-culture-map:
	python -m pip install -e culture_map

culture-map-help:
	python -m culture_map --help

qa-help:
	python qa_data/build_hf_dataset.py --help

pretraining-recipes:
	find src/step3_pretraining/3a_pretrain/continents -type f -name '*.md' | sort

clean-pycache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
