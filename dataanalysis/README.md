# Clinical record dashboard

Run the Streamlit dashboard from this directory:

```powershell
..\venv\Scripts\python.exe -m streamlit run csv_dashboard.py
```

`csv_dashboard.py` remains the stable entry point and runs `app.py`.

## Layout

| Directory | Responsibility |
| --- | --- |
| `core/` | Configuration, read-only PostgreSQL access, fetching, and the shared LLM client. |
| `extraction/` | Medication, laboratory, summary, and bounded AI extraction interfaces. |
| `services/` | Reviews, medication journeys, timeline, optional graph sync, and patient chat. |
| `storage/` | Local SQLite review and LLM cache. |
| `ui/` | Streamlit components and tab extension points. |
| `data/reference/` | Versioned deterministic clinical reference data. |

Legacy module names remain import shims while callers move to these packages.

## Data and safety

PostgreSQL is read-only for this dashboard. Review and LLM cache data are held locally in `data/cache.db`; Neo4j is optional and only syncs from the explicit **Sync graph** action. Credentials come from environment variables or the local `config` module and are never committed.

## Verification

```powershell
..\venv\Scripts\python.exe -m pytest -q tests
```
