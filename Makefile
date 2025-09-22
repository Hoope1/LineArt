format:
	ruff format .

lint:
	ruff check . --fix

typecheck:
	basedpyright --outputjson

vulture:
	vulture src tests --ignore-names 'use_sd,save_svg'

refurb:
	refurb src tests

deptry:
	deptry .

full-check:
	ruff format .
	ruff check . --fix
	basedpyright --outputjson
	vulture src tests --ignore-names 'use_sd,save_svg'
	refurb src tests
	deptry .

.PHONY: format lint typecheck vulture refurb deptry full-check
