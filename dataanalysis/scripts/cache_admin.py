#!/usr/bin/env python3
"""Cache administrative CLI for managing local SQLite storage.

Usage:
  python scripts/cache_admin.py status
  python scripts/cache_admin.py purge-expired
  python scripts/cache_admin.py purge-patient <patient_id>
  python scripts/cache_admin.py purge-all
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Ensure dataanalysis is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
ROOT_DIR = PROJECT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.ai_analysis_csv import _connect, purge_expired_cache, purge_patient, _DEFAULT_DB_PATH


def cmd_status(args):
    conn = _connect(args.db_path)
    try:
        rev_count = conn.execute("SELECT COUNT(*) AS c FROM ai_review").fetchone()["c"]
        cache_count = conn.execute("SELECT COUNT(*) AS c FROM llm_cache").fetchone()["c"]
        cache_patient_count = conn.execute("SELECT COUNT(*) AS c FROM llm_cache WHERE patient_hash IS NOT NULL").fetchone()["c"]
        faq_count = conn.execute("SELECT COUNT(*) AS c FROM chat_faq").fetchone()["c"]
        print("--- Local SQLite Cache Status ---")
        print(f"Database Path       : {args.db_path or _DEFAULT_DB_PATH}")
        print(f"AI Reviews          : {rev_count}")
        print(f"LLM Cache Total     : {cache_count}")
        print(f"LLM Cache (w/ hash) : {cache_patient_count}")
        print(f"Chat FAQ Entries    : {faq_count}")
        print("Note: Counts only displayed; no clinical content or patient IDs are output.")
    finally:
        conn.close()


def cmd_purge_expired(args):
    ttl_days = int(os.getenv("CACHE_TTL_DAYS", "0").strip() or "0")
    if ttl_days <= 0:
        print("CACHE_TTL_DAYS is 0 or unset; nothing to purge.")
        return
    deleted = purge_expired_cache(args.db_path)
    print(f"Purged {deleted} expired records (TTL: {ttl_days} days).")


def cmd_purge_patient(args):
    if not args.patient_id:
        print("Error: patient_id is required.")
        sys.exit(1)
    counts = purge_patient(args.patient_id, args.db_path)
    print(f"Purged records: ai_review={counts['ai_review']}, chat_faq={counts['chat_faq']}, llm_cache={counts['llm_cache']}")
    print("Note: Older unhashed cache rows can only be removed via purge-expired or purge-all.")


def cmd_purge_all(args):
    print("WARNING: This will permanently delete all records in ai_review, llm_cache, and chat_faq.")
    confirmation = input("Type 'DELETE ALL' to confirm: ").strip()
    if confirmation != "DELETE ALL":
        print("Purge aborted.")
        return
    conn = _connect(args.db_path)
    try:
        with conn:
            c1 = conn.execute("DELETE FROM ai_review").rowcount
            c2 = conn.execute("DELETE FROM llm_cache").rowcount
            c3 = conn.execute("DELETE FROM chat_faq").rowcount
        print(f"Purge complete: ai_review={c1}, llm_cache={c2}, chat_faq={c3} records deleted.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Cache Admin Utility (SQLite)")
    parser.add_argument("--db-path", default=None, help="Optional SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Display table row counts (no content)")
    subparsers.add_parser("purge-expired", help="Purge rows older than CACHE_TTL_DAYS")
    
    p_purge_patient = subparsers.add_parser("purge-patient", help="Purge all records for a specific patient")
    p_purge_patient.add_argument("patient_id", help="Patient identifier")

    subparsers.add_parser("purge-all", help="Purge all cache records (requires typed confirmation)")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "purge-expired":
        cmd_purge_expired(args)
    elif args.command == "purge-patient":
        cmd_purge_patient(args)
    elif args.command == "purge-all":
        cmd_purge_all(args)


if __name__ == "__main__":
    main()
