# Complete Architecture: Two-Stage System Pipeline

This system consists of two independent pipelines:
1. **Pipeline 1: Document Processing & Extraction Pipeline (`/`)** — Ingests raw clinical images, classifies them, extracts text/data via OCR, and writes structured records into PostgreSQL.
2. **Pipeline 2: Clinical Retrospective & Analytics Pipeline (`/dataanalysis`)** — Aggregates patient longitudinal history from PostgreSQL, optimizes AI usage using deterministic hashing & cache/delta logic, and serves a Streamlit dashboard.

---

## 1. Flowchart: Pipeline 1 (Ingestion, Classification & OCR)

```mermaid
flowchart TD
    Start([Run main.py]) --> Stage0[Stage 0: Batch Ingestion<br/>data_fetcher.fetch_batch]
    Stage0 --> PreFilter{Cached in filter.json?}
    
    PreFilter -- "Yes" --> CachedItems[Split by Cache Status]
    PreFilter -- "No" --> Download[Prefetch Images via ThreadPool]
    
    Download --> Stage1[Stage 1: CLIP & CV Classification<br/>classify_batch]
    Stage1 --> DocCheck{Is Document?}
    
    DocCheck -- "No" --> Exclude[Exclude non-clinical images]
    DocCheck -- "Yes" --> ClassifyType{Handwritten or Printed?}
    
    CachedItems --> SplitCombine[Combine Filtered Batches]
    ClassifyType --> SplitCombine
    
    SplitCombine -- "Printed Documents" --> Stage2[Stage 2: Mistral OCR<br/>run_printed_ocr]
    SplitCombine -- "Handwritten Documents" --> Stage3[Stage 3: EC2 Inference<br/>start_ec2_instance<br/>run_handwritten_ocr<br/>stop_ec2_instance]
    
    Stage2 --> DBWrite[(PostgreSQL Database<br/>visit, document_extraction, prescription)]
    Stage3 --> DBWrite
```

---

## 2. Flowchart: Pipeline 2 (Clinical Analysis, CSV Cache & Dashboard)

```mermaid
flowchart TD
    StartDash([User Enters Patient ID in Dashboard]) --> FetchDB[Fetch Patient History from DB<br/>analysis.fetch_patient_history<br/>LEFT JOIN LATERAL]
    FetchDB --> Hash[Compute SHA-256 Hash<br/>compute_source_hash]
    Hash --> CheckCSV{Check patient record in CSV<br/>get_patient_row}

    %% Case 1: First time analysis
    CheckCSV -- "No Record Found" --> FullRun1[Run Full AI Retrospective Analysis<br/>analyze_patient_with_ai]
    FullRun1 --> SaveNew[Insert Row in CSV<br/>mismatch_count = 0]
    SaveNew --> Render([Display Dashboard])

    %% Case 2: Unchanged cache hit
    CheckCSV -- "Record Exists" --> CompareHash{Current Hash == Stored Hash?}
    CompareHash -- "YES (Data Unchanged)" --> CacheHit[Load Cached Analysis from CSV<br/>Zero LLM Calls]
    CacheHit --> Render

    %% Case 3: Incremental vs Full re-run
    CompareHash -- "NO (Data Changed)" --> CheckThreshold{mismatch_count >= 4?}
    
    CheckThreshold -- "NO (< 4)" --> IncRun[Run Incremental AI Analysis<br/>Old Summary + New Visits Only<br/>analyze_patient_incremental]
    IncRun --> SaveInc[Update Row in CSV<br/>mismatch_count += 1]
    SaveInc --> Render

    CheckThreshold -- "YES (>= 4)" --> FullRun2[Run Full AI Analysis across All Visits<br/>analyze_patient_with_ai]
    FullRun2 --> SaveFull[Update Row in CSV<br/>mismatch_count = 0]
    SaveFull --> Render
```

---

## Module Overview

| Pipeline | Main Directory | Key Files | Function |
| :--- | :--- | :--- | :--- |
| **1. OCR Extraction** | Root (`/`) | `main.py`, `data/data_fetcher.py`, `preprocessing/clip_preprocessing.py`, `model/mistral_printed.py`, `model/handwritten.py` | Ingests images, filters non-documents, routes printed/handwritten to OCR engines, and stores extracted text in PostgreSQL. |
| **2. Analytics & Storage** | `dataanalysis/` | `analysis.py`, `Zai_analysis.py`, `storage/ai_analysis_csv.py`, `csv_dashboard.py` | Fetches consolidated visit history, tracks changes with SHA-256 hashes, maintains token-efficient CSV cache/upserts, and renders interactive dashboard. |





#	Group	Approx. patients	Typical diagnoses in your data
1	Skin / Hair	~7,600	eczema, tinea, xerosis, acne, scabies, melasma, urticaria, PMLE
2	Respiratory	~6,600	URTI, LRTI, cough/cold, flu, asthma, COPD
3	Musculoskeletal	~5,900	joint/knee pain, OA knee, LBA/LBP, DJD, spondylosis, myalgia
4	Cardiovascular (BP)	~4,000	HTN, HTN+T2DM, chest pain, CAD, dyslipidemia
5	Fever / Infection	~3,400	AFI, viral fever, herpes zoster, varicella
6	Gastrointestinal	~2,300	gastritis, gastroenteritis, GERD, IBS, pain abdomen, hemorrhoids
7	Diabetes / Thyroid	~2,100	T2DM, hypothyroidism
8	Weakness / Nutrition	~1,600	general weakness, anemia, nutritional
9	Neurological	~1,400	headache, vertigo, migraine, tension headache, neuropathy, radiculopathy
10	Women's health	~1,000	DUB, PCOS, amenorrhea, pregnancy, vulvovaginitis
11	Urinary / Sexual health	~500	UTI, STI
12	ENT	~420	CSOM, ear wax, pharyngitis, rhinitis
13	Injury / Wounds	~240	trauma, burn, abrasion
14	Mental health / Stress	~40	anxiety disorder
15	Eye / Dental	very few	none in the top diagnoses (shows up mostly in complaints)
About "tension": "tension headache" is in Neurological. The word "tension" on its own, meaning stress, goes in Mental health / Stress. Stress is rarely written as a diagnosis in your data, which is why that group is so small.

I also added a "Patient Type" card to the patient dashboard, just below the overview cards. It shows the patient's main group first, then the other groups with how many visits each appeared in. The new code hasn't been run or tested yet. The changes are: