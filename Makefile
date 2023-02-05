init:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	black .

lint-check:
	black --check .

freeze:
	pip freeze > requirements.txt