"""Optional Neo4j projection service facade."""
from neo4j_client import is_neo4j_configured, query_patient_graph, sync_patient_to_graph

__all__ = ["is_neo4j_configured", "query_patient_graph", "sync_patient_to_graph"]
