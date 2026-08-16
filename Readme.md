# OCR Pipeline

Fetches medical-document images from Drive, classifies each as printed or
handwritten with CLIP, routes printed images to PaddleOCR and handwritten
images to an EC2-hosted vLLM model, and merges both results into a CSV.

```
data/data_fetcher.py   -> pull a limited, resumable batch (Drive today, swappable later)
        |
        v
main.py (STAGE 1)      -> CLIP classify -> output/filter.json
        |
        +---------------------------+
        v                           v
model/paddel.py             model/handwritten.py
(printed)                   (handwritten, via EC2)
        |                           |
        v                           v
output/output.json          output/result.json
        \___________________________/
                    |
                    v
        main.py merge_to_csv() -> output/final_report.csv
```

## Why three separate preprocessing modules

`preprocessing/clip_preprocessing.py`, `printed_preprocessing.py`, and
`handwritten_preprocessing.py` are deliberately not shared code. Each model
stage resizes and formats the image differently, and each has its own
confidence thresholds in `config.py`:

| Stage | Max dimension | Threshold(s) | Why different |
|---|---|---|---|
| CLIP classifier | 2000px (source), tiled 3x2 | `HANDWRITTEN_THRESHOLD=0.65`, tile-vote ratio `0.75`, structural table override | Tile-vote based; a printed form with handwritten fill-ins should still classify as printed |
| PaddleOCR (printed) | 2000px | `PADDLE_CONFIDENCE_THRESHOLD=0.5` (`0.35` for long text) | OCR recognition score, not a classification probability |
| Handwritten (EC2 vLLM) | 720px | none (retry/timeout instead) | VLM has no per-token confidence score to gate on; size cap is about latency/cost, not accuracy |

Never reuse one stage's threshold for another -- they measure different
things on differently-sized images.

## Setup

```bash
cp .env.example .env   # fill in DRIVE_API_KEY, DRIVE_ROOT_FOLDER_ID, etc.
pip install -r requirements.txt --break-system-packages
# then install the GPU build of paddlepaddle for your CUDA version -- see
# the comment at the top of requirements.txt. Do not install the CPU wheel.
```

## Run

```bash
python main.py
```

Resumable: images already present in `output/output.json` or
`output/result.json` are skipped by `data/data_fetcher.load_skip_keys()`
before anything is even downloaded. `output/filter.json` caches CLIP
decisions per image, keyed to a fingerprint of the current thresholds --
change any threshold in `config.py` and affected images get reclassified
automatically.

## EC2 for the handwritten model

Two ways to run it:

1. **Manual (default)** -- open your own SSH tunnel to the instance, leave
   `AUTO_MANAGE_EC2=false` in `.env`, and `main.py` just calls the endpoint.
   ```bash
   ssh -i "path/to/handwritten.pem" -N -L 8000:127.0.0.1:8000 ec2-user@<ip>
   ```
2. **Automatic** -- set `AUTO_MANAGE_EC2=true` and `EC2_INSTANCE_ID` in
   `.env`. `main.py` will start the instance, poll `/health` until the model
   server is up, run the handwritten batch, and stop the instance again
   when done (`model/load_model.start_ec2_instance` / `stop_ec2_instance`).

## Future work (not yet implemented, hooks left in place)

- **Swap data source**: `config.DATA_SOURCE` and `data/data_fetcher.build_manifest`
  already branch on it; add a `local`/`s3` implementation there.
- **Patient visit timeline**: once `final_report.csv` has multiple rows per
  patient folder, group by `folder` and flag folders with 3-4+ visits for a
  timeline/analysis view. Not built yet -- needs a "visit" concept the current
  JSON schema doesn't capture (currently one row per image, not per visit).
- **Streamlit dashboard**: read `final_report.csv` (or `output.json`/`result.json`
  directly) filtered by patient/folder ID. Not built yet.
- **Full agent/rule-based orchestration**: `main.py` already runs stages 1-4
  in order and auto start/stops EC2 around stage 3 when `AUTO_MANAGE_EC2=true`.
  Turning this into an agent-driven or fully rule-based scheduler (e.g. cron,
  Airflow, or an LLM-agent loop) is the next step, not implemented here.
- **CSV merge rules**: `main.merge_to_csv()` currently does a straightforward
  rule-based flatten (one row per image) and calls a no-op `llm_refine_row()`
  hook. Plug in the real reconciliation/summarization rules and the LLM call
  once they're defined.

## Constraints 

- Images already in `output.json`/`result.json` are never re-downloaded or
  re-run (`data_fetcher.load_skip_keys`).
- JSON output shape is `{folder: {image: [...]}}` everywhere (`output.json`,
  `result.json`, `filter.json`).
- PaddleOCR runs GPU-only (`config.PADDLE_DEVICE`, GPU-only wheel noted in
  `requirements.txt`).



1. fetch assigned limited data from the data source /data folder  currently from drive 
2. Then i have to pass it with classification model clip then generate filter.json
3. I have two diff model see classification then run for printed first through paddel and save to output.json.
4. once paddel is done then i have handwritten model in ec2 it will  give an output to start ec2 model or it will start ec2 model pass handwritten images to it and generate result.json from it.
5. once everything is done it will create csv from both the json by some rule based and llm.
6. future work 
 -- I can change data source 
 -- use table to see patient visit if more then 3-4 visit will create a analysis and time line based represention of patient condition
 -- can add simple streamlit dashboard to see result based on perticular patient id
 -- connect everything in pipeline via agents or simple rule based and automettically turning on the ec2 run save in result.json then turning   in off

7. constraint
   -- skip images which is already present in output.json or result.json
   -- json structure 
   {
    foldername{
        image1{....},
        image2{.....}, // ocr
    }
    folder2{
        ...
    }
   }
   -- run downloaded model like paddelocr on gpu , pip install only gpu version
   

