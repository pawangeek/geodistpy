lint:
	poetry run black . --check
coverage:
	poetry run pytest --cov --cov-config=.coveragerc --cov-report xml --cov-report term-missing:skip-covered
test:
	poetry run pytest