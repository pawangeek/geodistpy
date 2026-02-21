lint:
	poetry run black . --check
coverage:
	poetry run pytest --cov --cov-report xml --cov-report term-missing:skip-covered --junitxml=junit.xml
test:
	poetry run pytest