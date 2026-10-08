"""Neo4j graph synchronization and querying layer.

Uses environment variables:
- NEO4J_URI
- NEO4J_USERNAME
- NEO4J_PASSWORD
- NEO4J_DATABASE (optional, defaults to 'neo4j')

If Neo4j is not configured or connection fails, gracefully falls back
without crashing the application.
"""
import logging
import os

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    _NEO4J_INSTALLED = True
except ImportError:
    GraphDatabase = None
    _NEO4J_INSTALLED = False


def is_neo4j_configured() -> bool:
    """Return True only if credentials are provided in environment variables."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    return bool(_NEO4J_INSTALLED and uri and user and password)


def get_driver():
    """Create and return a Neo4j driver, or None if unavailable/unconfigured."""
    if not is_neo4j_configured():
        return None
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver
    except Exception as exc:
        logger.warning("Failed to initialize Neo4j driver: %s", exc)
        return None


def sync_patient_to_graph(patient_data) -> bool:
    """Synchronize a single patient's clinical record into Neo4j using idempotent MERGE queries.

    Nodes:
    (:Patient {patient_id})
    (:Visit {patient_id, visit_id, date, chief_complaint})
    (:Diagnosis {patient_id, name, type})
    (:Medication {patient_id, name, dosage, frequency, source})
    (:Lab {patient_id, test_name, value, reference, status})
    (:FollowUp {patient_id, scheduled_date, status})
    (:Document {patient_id, visit_id, document_id, label, date})
    (:DocumentChunk {patient_id, visit_id, document_id, chunk_id, text, date, doc_type})

    Relationships:
    Patient-[:HAS_VISIT]->Visit
    Visit-[:HAS_DIAGNOSIS]->Diagnosis
    Visit-[:HAS_MEDICATION]->Medication
    Visit-[:HAS_LAB]->Lab
    Visit-[:HAS_FOLLOWUP]->FollowUp
    Visit-[:HAS_DOCUMENT]->Document
    Document-[:HAS_CHUNK]->DocumentChunk
    """
    if not patient_data:
        return False
    driver = get_driver()
    if not driver:
        return False

    patient_id = str(patient_data.get("patient_id", ""))
    if not patient_id:
        return False

    database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB_NAME") or "neo4j"
    visits = patient_data.get("visits", [])

    try:
        with driver.session(database=database) as session:
            # 1. Merge Patient
            session.run(
                "MERGE (p:Patient {patient_id: $patient_id}) "
                "SET p.total_visits = $total_visits",
                patient_id=patient_id,
                total_visits=patient_data.get("total_visits", len(visits)),
            )

            # 2. Iterate Visits
            for idx, v in enumerate(visits, start=1):
                v_id = str(v.get("visit_id") or f"visit_{idx}")
                v_date = (v.get("visit_date") or "Unknown")[:10]
                cc = str(v.get("db_chief_complaint") or v.get("chief_complaint") or "")

                session.run(
                    """
                    MATCH (p:Patient {patient_id: $patient_id})
                    MERGE (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                    SET v.date = $date, v.chief_complaint = $chief_complaint,
                        v.provisional_diagnosis = $provisional_diagnosis,
                        v.confirmed_diagnosis = $confirmed_diagnosis,
                        v.vitals_json = $vitals_json
                    MERGE (p)-[:HAS_VISIT]->(v)
                    """,
                    patient_id=patient_id,
                    visit_id=v_id,
                    date=v_date,
                    chief_complaint=cc,
                    provisional_diagnosis=str(v.get("db_provisional_dx") or v.get("provisional_diagnosis") or ""),
                    confirmed_diagnosis=str(v.get("db_confirmed_dx") or v.get("confirmed_diagnosis") or ""),
                    vitals_json=str(v.get("vitals") or v.get("db_vitals") or ""),
                )

                # Diagnoses (Provisional & Confirmed)
                dx_tuples = []
                for dx_name, dx_type in [
                    (v.get("db_confirmed_dx"), "confirmed"),
                    (v.get("confirmed_diagnosis"), "confirmed"),
                    (v.get("db_provisional_dx"), "provisional"),
                    (v.get("provisional_diagnosis"), "provisional"),
                ]:
                    if dx_name and str(dx_name).strip() and str(dx_name).strip().lower() not in ("—", "-", "none"):
                        dx_tuples.append((str(dx_name).strip(), dx_type))

                for name, dx_type in dx_tuples:
                    session.run(
                        """
                        MATCH (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                        MERGE (d:Diagnosis {patient_id: $patient_id, name: $name, type: $dx_type})
                        MERGE (v)-[:HAS_DIAGNOSIS]->(d)
                        """,
                        patient_id=patient_id,
                        visit_id=v_id,
                        name=name,
                        dx_type=dx_type,
                    )

                # Medications (Structured Prescriptions)
                for rx in v.get("db_medications", []) or []:
                    m_name = str(rx.get("name", "")).strip()
                    if m_name:
                        session.run(
                            """
                            MATCH (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                            MERGE (m:Medication {patient_id: $patient_id, name: $name, dosage: $dosage, frequency: $frequency, source: 'prescription'})
                            MERGE (v)-[:HAS_MEDICATION]->(m)
                            """,
                            patient_id=patient_id,
                            visit_id=v_id,
                            name=m_name,
                            dosage=str(rx.get("dosage", "")),
                            frequency=str(rx.get("frequency", "")),
                        )

                # Labs
                for lab in v.get("lab_results", []) or []:
                    t_name = str(lab.get("test_name", "")).strip()
                    if t_name:
                        session.run(
                            """
                            MATCH (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                            MERGE (l:Lab {patient_id: $patient_id, test_name: $test_name, value: $value, reference: $reference, status: $status})
                            MERGE (v)-[:HAS_LAB]->(l)
                            """,
                            patient_id=patient_id,
                            visit_id=v_id,
                            test_name=t_name,
                            value=str(lab.get("value", "")),
                            reference=str(lab.get("reference", "")),
                            status=str(lab.get("status", "")),
                        )

                # Follow-ups
                for fu in v.get("followups", []) or []:
                    f_date = str(fu.get("scheduled_date", ""))
                    f_status = str(fu.get("status", ""))
                    session.run(
                        """
                        MATCH (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                        MERGE (fu:FollowUp {patient_id: $patient_id, scheduled_date: $scheduled_date, status: $status})
                        MERGE (v)-[:HAS_FOLLOWUP]->(fu)
                        """,
                        patient_id=patient_id,
                        visit_id=v_id,
                        scheduled_date=f_date,
                        status=f_status,
                    )

                # Documents & Chunks
                for doc in v.get("documents", []) or []:
                    doc_id = str(doc.get("doc_id", "doc_unknown"))
                    doc_label = str(doc.get("document_label", "Clinical Document"))
                    session.run(
                        """
                        MATCH (v:Visit {patient_id: $patient_id, visit_id: $visit_id})
                        MERGE (d:Document {patient_id: $patient_id, visit_id: $visit_id, document_id: $document_id})
                        SET d.label = $label, d.date = $date
                        MERGE (v)-[:HAS_DOCUMENT]->(d)
                        """,
                        patient_id=patient_id,
                        visit_id=v_id,
                        document_id=doc_id,
                        label=doc_label,
                        date=v_date,
                    )

                    # Chunks from document extracted text
                    extracted_lines = doc.get("extracted_text", []) or []
                    for c_idx, line in enumerate(extracted_lines[:15], start=1):
                        line_str = str(line).strip()
                        if line_str:
                            chunk_id = f"{doc_id}_c{c_idx}"
                            session.run(
                                """
                                MATCH (d:Document {patient_id: $patient_id, visit_id: $visit_id, document_id: $document_id})
                                MERGE (c:DocumentChunk {patient_id: $patient_id, visit_id: $visit_id, document_id: $document_id, chunk_id: $chunk_id})
                                SET c.text = $text, c.date = $date, c.doc_type = $doc_type
                                MERGE (d)-[:HAS_CHUNK]->(c)
                                """,
                                patient_id=patient_id,
                                visit_id=v_id,
                                document_id=doc_id,
                                chunk_id=chunk_id,
                                text=line_str,
                                date=v_date,
                                doc_type=doc_label,
                            )
        return True
    except Exception as exc:
        logger.warning("Neo4j synchronization error for patient %s: %s", patient_id, exc)
        return False
    finally:
        driver.close()


def query_patient_graph(patient_id: str, query_type: str = "all", limit: int = 100) -> list:
    """Retrieve patient-scoped graph nodes, including nodes linked by legacy relationships."""
    if not is_neo4j_configured():
        return []
    driver = get_driver()
    if not driver:
        return []

    database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB_NAME") or "neo4j"
    results = []

    try:
        with driver.session(database=database) as session:
            # Match direct patient_id properties (including orphaned legacy nodes)
            # and nodes connected to this patient's visits (legacy FollowUp nodes).
            cypher = """
            MATCH (n)
            WHERE (n.patient_id IS NOT NULL AND toString(n.patient_id) = $patient_id)
               OR (n:FollowUp AND EXISTS {
                    MATCH (:Visit {patient_id: $patient_id})-[:HAS_FOLLOWUP]->(n)
                  })
            WITH DISTINCT n
            WHERE $query_type = 'all' OR $query_type IN labels(n)
            OPTIONAL MATCH (v:Visit {patient_id: $patient_id})-[*1..2]->(n)
            RETURN labels(n) AS labels, properties(n) AS props,
                   v.visit_id AS visit_id, v.date AS date
            LIMIT $limit
            """
            cursor = session.run(
                cypher, patient_id=str(patient_id), query_type=query_type,
                limit=max(1, int(limit)),
            )
            for record in cursor:
                row = dict(record)
                props = row.get("props") or {}
                labels = row.get("labels") or []
                category = next((name for name in labels if name != "Patient"), "Patient record")
                if category == "Patient":
                    continue
                # Include all source fields so the model can answer detail questions
                # (dose, reference range, status, units, dates) without guessing.
                content = "; ".join(f"{key}: {value}" for key, value in props.items() if value not in (None, ""))
                results.append({
                    "category": category,
                    "content": content,
                    "source_id": props.get("document_id") or props.get("visit_id") or props.get("chunk_id") or props.get("test_name") or props.get("name") or category,
                    "visit_id": row.get("visit_id") or props.get("visit_id") or "Unknown",
                    "date": row.get("date") or props.get("date") or props.get("scheduled_date") or "Date not recorded",
                })
    except Exception as exc:
        logger.warning("Neo4j query error for patient %s: %s", patient_id, exc)
        return []
    finally:
        driver.close()

    return results
