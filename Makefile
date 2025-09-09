format:
	ruff format .

lint:
	ruff check . --fix

typecheck:
	basedpyright --outputjson

vulture:
	vulture src tests

refurb:
	refurb src tests

deptry:
	deptry .

full-check:
	ruff format .
	ruff check . --fix
	basedpyright --outputjson
	vulture src tests
	refurb src tests
	deptry .

.PHONY: format lint typecheck vulture refurb deptry full-check
