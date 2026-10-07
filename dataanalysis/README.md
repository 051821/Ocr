# Clinical AI Retrospective Analysis & CSV Persistence Pipeline

## Pipeline Flowchart

```mermaid
flowchart TD
    Start([User Selects Patient ID]) --> Fetch[Fetch Patient History from DB<br/>fetch_patient_history]
    Fetch --> Hash[Compute SHA-256 Hash of Source Data<br/>compute_source_hash]
    Hash --> CheckCSV{Check CSV Record for Patient<br/>get_patient_row}

    %% Branch 1: No previous entry
    CheckCSV -- "No Record Found" --> FullRun1[Run Full AI Analysis<br/>analyze_patient_with_ai]
    FullRun1 --> SaveNew[Save New Row in CSV<br/>upsert_ai_analysis<br/>mismatch_count = 0]
    SaveNew --> Display([Display AI Analysis])

    %% Branch 2: Hash Matches
    CheckCSV -- "Record Exists" --> CompareHash{Compare Hashes<br/>current_hash == stored_hash?}
    CompareHash -- "YES (Data Unchanged)" --> CacheHit[Load Cached Analysis from CSV<br/>Zero LLM Calls]
    CacheHit --> Display

    %% Branch 3: Hash Mismatch
    CompareHash -- "NO (Data Changed)" --> CheckCount{mismatch_count >= 4?}
    
    CheckCount -- "NO (< 4)" --> IncrementalRun[Run Incremental AI Analysis<br/>Old Summary + New Visits Only<br/>analyze_patient_incremental]
    IncrementalRun --> RewriteInc[Rewrite Row in CSV<br/>mismatch_count += 1<br/>upsert_ai_analysis]
    RewriteInc --> Display

    CheckCount -- "YES (>= 4)" --> FullRun2[Run Full AI Analysis on All Visits<br/>analyze_patient_with_ai]
    FullRun2 --> RewriteFull[Rewrite Row in CSV<br/>mismatch_count = 0<br/>upsert_ai_analysis]
    RewriteFull --> Display
```

---

## File Roles & Architecture

| File | Purpose |
| :--- | :--- |
| **`analysis.py`** | Fetches visit history, aggregated documents, and prescriptions using PostgreSQL `LEFT JOIN LATERAL`. |
| **`Zai_analysis.py`** | Manages full prompt (`analyze_patient_with_ai`) and token-saving delta prompt (`analyze_patient_incremental`). |
| **`storage/ai_analysis_csv.py`** | Handles single-row per patient CSV upserts, hash computation, and JSON storage in `data/patient_ai_analysis.csv`. |
| **`csv_dashboard.py`** | Streamlit UI executing the automatic cache / incremental / full refresh evaluation flowchart. |
