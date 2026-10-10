"""Compact, patient-scoped Neo4j projection for the clinical record.

Neo4j is an optional read model. PostgreSQL remains the source of record and
is never written by this module.
"""
import json
import logging
import os
import re
from threading import Lock

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    _NEO4J_INSTALLED = True
except ImportError:
    GraphDatabase = None
    _NEO4J_INSTALLED = False

_DRIVER = None
_DRIVER_LOCK = Lock()
_CONSTRAINTS = (
    "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT visit_id_unique IF NOT EXISTS FOR (v:Visit) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT dx_name_unique IF NOT EXISTS FOR (d:Dx) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT lab_test_name_unique IF NOT EXISTS FOR (l:LabTest) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Doc) REQUIRE d.id IS UNIQUE",
)


def is_neo4j_configured() -> bool:
    return bool(_NEO4J_INSTALLED and os.getenv("NEO4J_URI") and
                (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")) and
                (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")))


def get_driver():
    """Return the process-wide Neo4j driver; it is not recreated per request."""
    global _DRIVER
    if not is_neo4j_configured():
        return None
    with _DRIVER_LOCK:
        if _DRIVER is None:
            try:
                _DRIVER = GraphDatabase.driver(
                    os.environ["NEO4J_URI"],
                    auth=(os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER"),
                          os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")),
                )
            except Exception as exc:
                logger.warning("Failed to initialize Neo4j driver: %s", exc)
                return None
    return _DRIVER


def _clean(value):
    return str(value or "").strip()


def _normalized(value):
    return " ".join(_clean(value).casefold().split())


def _vitals(visit):
    raw = visit.get("db_vitals_json") or visit.get("vitals") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {"raw": raw}
    raw = raw if isinstance(raw, dict) else {}
    values = {str(key).casefold(): value for key, value in raw.items()}
    bp = _clean(values.get("bp") or values.get("blood_pressure") or values.get("blood pressure"))
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", bp)
    return {
        "bp_sys": int(match.group(1)) if match else None,
        "bp_dia": int(match.group(2)) if match else None,
        "pulse": _clean(values.get("pulse") or values.get("heart_rate") or values.get("heart rate")) or None,
        "temp": _clean(values.get("temperature") or values.get("temp")) or None,
        "spo2": _clean(values.get("spo2") or values.get("oxygen_saturation")) or None,
        "wt": _clean(values.get("weight") or values.get("wt")) or None,
    }


def _followups(visit):
    return ["; ".join(filter(None, (_clean(item.get("scheduled_date")), _clean(item.get("status")))))
            for item in visit.get("followups", []) or [] if isinstance(item, dict)]


def _build_graph_payload(patient_data):
    """Convert a record into compact rows. No OCR chunks are included."""
    patient_id = _clean(patient_data.get("patient_id"))
    visits, rx_rows, dx_rows, lab_rows, doc_rows = [], [], [], [], []
    for ordinal, visit in enumerate(patient_data.get("visits", []) or [], start=1):
        visit_ref = _clean(visit.get("visit_id")) or str(ordinal)
        visit_id = f"{patient_id}:{visit_ref}"
        provisional = _clean(visit.get("db_provisional_dx") or visit.get("provisional_diagnosis"))
        confirmed = _clean(visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis"))
        row = {
            "id": visit_id, "date": _clean(visit.get("visit_date"))[:10] or "Unknown",
            "cc": _clean(visit.get("db_chief_complaint") or visit.get("chief_complaint")),
            "dx_prov": provisional, "dx_conf": confirmed,
            "followups": _followups(visit), **_vitals(visit),
        }
        visits.append(row)
        for name, kind in ((provisional, "provisional"), (confirmed, "confirmed")):
            normalized = _normalized(name)
            if normalized and normalized not in {"-", "none", "not documented"}:
                dx_rows.append({"visit_id": visit_id, "name": normalized, "type": kind})
        for medicine in visit.get("db_medications", []) or []:
            name = _normalized(medicine.get("name"))
            if name:
                rx_rows.append({"visit_id": visit_id, "name": name,
                                "dose": _clean(medicine.get("dosage") or medicine.get("dose")),
                                "freq": _clean(medicine.get("frequency") or medicine.get("freq")),
                                "dur": _clean(medicine.get("duration"))})
        for lab in visit.get("lab_results", []) or []:
            name = _normalized(lab.get("test_name") or lab.get("name"))
            if name:
                number = re.search(r"[-+]?\d+(?:\.\d+)?", _clean(lab.get("value")))
                lab_rows.append({"visit_id": visit_id, "name": name, "val": _clean(lab.get("value")),
                                 "num": float(number.group()) if number else None,
                                 "ref": _clean(lab.get("reference") or lab.get("reference_range")),
                                 "status": _clean(lab.get("status"))})
        for doc_num, doc in enumerate((visit.get("documents", []) or [])[:1], start=1):
            doc_ref = _clean(doc.get("doc_id")) or str(doc_num)
            summary = _clean(doc.get("clinical_summary") or doc.get("summary"))
            if not summary:
                summary = " ".join(_clean(line) for line in (doc.get("extracted_text", []) or [])[:3])
            doc_rows.append({"id": f"{patient_id}:{visit_ref}:{doc_ref}", "visit_id": visit_id,
                             "patient_id": patient_id, "label": _clean(doc.get("document_label")) or "Clinical document",
                             "date": row["date"], "summary": summary[:300]})
    return {"patient_id": patient_id, "source_hash": _clean(patient_data.get("_review_source_hash")),
            "total_visits": len(visits), "visits": visits, "rx": rx_rows, "dx": dx_rows,
            "labs": lab_rows, "docs": doc_rows}


def _ensure_constraints(session):
    for statement in _CONSTRAINTS:
        session.run(statement).consume()


def _replace_patient_tx(tx, payload):
    """Rewrite only this patient's visit subtree in one Neo4j transaction."""
    patient_id = payload["patient_id"]
    tx.run("MATCH (d:Doc {patient_id:$patient_id}) DETACH DELETE d", patient_id=patient_id).consume()
    tx.run("MATCH (:Patient {id:$patient_id})-[:HAS_VISIT]->(v:Visit) DETACH DELETE v", patient_id=patient_id).consume()
    tx.run("MERGE (p:Patient {id:$patient_id}) SET p.src_hash=$source_hash, p.total_visits=$total_visits",
           patient_id=patient_id, source_hash=payload["source_hash"], total_visits=payload["total_visits"]).consume()
    tx.run("""
        UNWIND $rows AS row
        MATCH (p:Patient {id:$patient_id})
        MERGE (v:Visit {id:row.id})
        SET v += row
        MERGE (p)-[:HAS_VISIT]->(v)
    """, patient_id=patient_id, rows=payload["visits"]).consume()
    tx.run("""
        UNWIND $rows AS row
        MATCH (v:Visit {id:row.visit_id})
        MERGE (d:Drug {name:row.name})
        MERGE (v)-[r:RX]->(d) SET r.dose=row.dose, r.freq=row.freq, r.dur=row.dur
    """, rows=payload["rx"]).consume()
    tx.run("""
        UNWIND $rows AS row
        MATCH (v:Visit {id:row.visit_id})
        MERGE (d:Dx {name:row.name})
        MERGE (v)-[r:DX]->(d) SET r.type=row.type
    """, rows=payload["dx"]).consume()
    tx.run("""
        UNWIND $rows AS row
        MATCH (v:Visit {id:row.visit_id})
        MERGE (l:LabTest {name:row.name})
        MERGE (v)-[r:LAB]->(l) SET r.val=row.val, r.num=row.num, r.ref=row.ref, r.status=row.status
    """, rows=payload["labs"]).consume()
    tx.run("""
        UNWIND $rows AS row
        MATCH (v:Visit {id:row.visit_id})
        MERGE (d:Doc {id:row.id}) SET d += row
        MERGE (v)-[:HAS_DOC]->(d)
    """, rows=payload["docs"]).consume()


def sync_patient_to_graph(patient_data) -> bool:
    """Synchronize on an explicit user action; skip when the source hash matches."""
    if not patient_data:
        return False
    payload = _build_graph_payload(patient_data)
    if not payload["patient_id"] or not payload["source_hash"]:
        logger.warning("Neo4j sync skipped because patient id or source hash is unavailable")
        return False
    driver = get_driver()
    if not driver:
        return False
    database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB_NAME") or "neo4j"
    try:
        with driver.session(database=database) as session:
            existing = session.run("MATCH (p:Patient {id:$patient_id}) RETURN p.src_hash AS src_hash",
                                   patient_id=payload["patient_id"]).single()
            if existing and existing.get("src_hash") == payload["source_hash"]:
                return True
            _ensure_constraints(session)
            session.execute_write(_replace_patient_tx, payload)
        return True
    except Exception as exc:
        logger.warning("Neo4j synchronization error for patient %s: %s", payload["patient_id"], exc)
        return False


def query_patient_graph(patient_id: str, limit: int = 100) -> list:
    """Read only graph facts reachable through the requested patient's visits."""
    driver = get_driver()
    if not driver:
        return []
    database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB_NAME") or "neo4j"
    cypher = """
        MATCH (:Patient {id:$patient_id})-[:HAS_VISIT]->(v:Visit)
        OPTIONAL MATCH (v)-[r]->(node)
        WHERE type(r) IN ['RX', 'DX', 'LAB', 'HAS_DOC']
        RETURN properties(v) AS visit, type(r) AS relation, labels(node) AS labels,
               properties(node) AS node, properties(r) AS relationship
        LIMIT $limit
    """
    try:
        with driver.session(database=database) as session:
            rows = session.run(cypher, patient_id=str(patient_id), limit=max(1, int(limit)))
            output = []
            for record in rows:
                row = dict(record)
                visit, node, relationship = row.get("visit") or {}, row.get("node") or {}, row.get("relationship") or {}
                content = "; ".join(f"{key}: {value}" for key, value in {**visit, **node, **relationship}.items() if value not in (None, "", []))
                if content:
                    output.append({"category": row.get("relation") or "visit", "content": content,
                                   "source_id": visit.get("id", "Unknown"), "visit_id": visit.get("id", "Unknown"),
                                   "date": visit.get("date", "Date not recorded")})
            return output
    except Exception as exc:
        logger.warning("Neo4j query error for patient %s: %s", patient_id, exc)
        return []
