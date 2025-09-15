.PHONY: env data bench-amazon bench-yelp bench-taobao tables test clean

env:
	conda env create -f environment.yml || true

data:
	python scripts/download_data.py --datasets flickr amazon yelp taobao

bench-amazon:
	python scripts/benchmark.py --config configs/experiments/amazon_table2.yaml

bench-yelp:
	python scripts/benchmark.py --config configs/experiments/yelp_table3.yaml

bench-taobao:
	python scripts/benchmark.py --config configs/experiments/taobao_table4.yaml

tables:
	python scripts/make_tables.py

test:
	pytest -q

clean:
	rm -rf outputs/* __pycache__ */__pycache__
