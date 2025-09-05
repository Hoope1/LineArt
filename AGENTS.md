Nach dem nach dem Programmieren, ändern, korrigieren, ergänzen des Codes sollst du eine saubere, reproduzierbare Qualitäts-Pipeline fahren – schnell zuerst, tiefgründig danach:
Also immer vor dem Abschluss!
1. Schnelle Auto-Korrektur & Formatierung

Ruff (lint + quick-fix + Imports):
ruff check . --fix
Fängt Syntax-/Style-Fehler (E/F/W …), sortiert Importe (Regelgruppe I), kann auch Modernisierungen (UP/pyupgrade-Regeln) und Docstring-Checks (D) übernehmen – extrem schnell. 

Black (finale Formatierung):
black .
Erzwingt einheitlichen Stil, keine Diskussionen über Formatdetails. (Black liest optional deine pyproject.toml-Settings.) 

Optional isort: nur wenn du nicht willst, dass Ruff die Importe sortiert: isort . (Ruff kann das bereits.) 



2. Statische Typprüfung (zwei Perspektiven, mehr Treffer)

basedpyright (strenger, sehr schnell): basedpyright
Vorteil: reines PyPI-Paket, keine Node-Abhängigkeit. 

mypy (zweite Meinung, anderes Regelset): mypy
Beide Checker finden unterschiedliche Klassen von Typ-Fehlern; Konfiguration bequem in pyproject.toml unter [tool.mypy]. 



3. Tiefenanalyse der Codequalität

Pylint: pylint src/
Deckt Naming-Konventionen, Code-Smells, Refactoring-Hinweise ab; findet anderes als Ruff. Pylint kann pyproject.toml lesen. 

Vulture: vulture src/
Sucht toten/ungenutzten Code (Funktionen, Variablen, Klassen). Konfigurierbar via [tool.vulture] in pyproject.toml. 



4. Abhängigkeits-Hygiene

Deptry: deptry .
Findet fehlende, ungenutzte, transitive und falsch genutzte Stdlib-Im-porte; versteht PEP 621-Metadaten aus pyproject.toml ([project], [project.optional-dependencies]). Konfigurierbar via [tool.deptry]. 



5. Dokumentationsdisziplin

pydocstyle: pydocstyle src/
Prüft PEP 257-konforme Docstrings. (Alternativ kannst du die D-Regeln über Ruff laufen lassen und pydocstyle weglassen.) 





---

Minimal sinnvolle Reihenfolge (CLI)

# 1) Schnell fixen & formatieren
ruff check . --fix
black .

# 2) Typchecks
basedpyright
mypy

# 3) Tiefenanalyse
pylint src/
vulture src/

# 4) Dependencies prüfen
deptry .

# 5) (optional, falls nicht via Ruff-D-Regeln)
pydocstyle src/
