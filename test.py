#!/usr/bin/env python3
"""
ec2_run_direct.py

Fill in the CONFIG block below and just run:
    python3 ec2_run_direct.py

No CLI args, no aws configure, no .env -- everything is a variable in this
file. Starts the instance, "does work" for RUN_SECONDS, then stops it --
guaranteed via try/finally even if something crashes mid-run.

*** SECURITY WARNING ***
Hardcoding AWS keys in a .py file is fine for a quick local test against a
dummy/throwaway instance, but:
  - never commit this file with real keys filled in (add it to .gitignore)
  - use a dedicated IAM user scoped to just this one instance (see the
    ec2:StartInstances / ec2:StopInstances / ec2:DescribeInstances policy
    from earlier), never your root/admin keys
  - delete/rotate the key when you're done testing
For anything beyond a throwaway test, switch back to `aws configure` or
environment variables so the keys never sit in a file at all.
"""
import sys
import time

import boto3

# ============================================================
# CONFIG -- fill these in
# ============================================================
AWS_ACCESS_KEY_ID = "PASTE_YOUR_ACCESS_KEY_ID_HERE"
AWS_SECRET_ACCESS_KEY = "PASTE_YOUR_SECRET_ACCESS_KEY_HERE"
AWS_REGION = "ap-south-1"

INSTANCE_ID = "i-0123456789abcdef0"

RUN_SECONDS = 120        # how long to "work" before stopping the instance
BOOT_WAIT_SECONDS = 90   # sleep after instance reaches "running" before starting work
STOP_WAIT = True         # block until instance is fully stopped before exiting

# optional: set to a real URL (e.g. "http://<ip>:8000/health") once you're
# testing against an instance that actually runs the model server.
# Leave as None for a dummy instance with nothing listening on it.
HEALTH_URL = None
HEALTH_POLL_SECONDS = 10
HEALTH_TIMEOUT_SECONDS = 600
# ============================================================


def get_client():
    return boto3.client(
        "ec2",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_state(client, instance_id):
    resp = client.describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0]["State"]["Name"]


def start_instance(client):
    state = get_state(client, INSTANCE_ID)
    if state == "running":
        print(f"[ec2] {INSTANCE_ID} already running.")
        return

    print(f"[ec2] {INSTANCE_ID} state={state}, starting...")
    client.start_instances(InstanceIds=[INSTANCE_ID])
    client.get_waiter("instance_running").wait(InstanceIds=[INSTANCE_ID])
    print(f"[ec2] {INSTANCE_ID} is now running.")

    resp = client.describe_instances(InstanceIds=[INSTANCE_ID])
    inst = resp["Reservations"][0]["Instances"][0]
    print(f"[ec2] public_ip={inst.get('PublicIpAddress')} private_ip={inst.get('PrivateIpAddress')}")

    print(f"[ec2] waiting {BOOT_WAIT_SECONDS}s for the OS/services to come up...")
    time.sleep(BOOT_WAIT_SECONDS)


def stop_instance(client):
    state = get_state(client, INSTANCE_ID)
    if state == "stopped":
        print(f"[ec2] {INSTANCE_ID} already stopped.")
        return

    print(f"[ec2] {INSTANCE_ID} state={state}, stopping...")
    client.stop_instances(InstanceIds=[INSTANCE_ID])
    if STOP_WAIT:
        client.get_waiter("instance_stopped").wait(InstanceIds=[INSTANCE_ID])
        print(f"[ec2] {INSTANCE_ID} is now stopped.")
    else:
        print("[ec2] stop requested (not waiting for confirmation).")


def wait_for_endpoint_ready():
    import requests

    print(f"[health] waiting for {HEALTH_URL} to respond...")
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                print("[health] endpoint is ready.")
                return True
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_SECONDS)
    print(f"[health] endpoint did not become ready within {HEALTH_TIMEOUT_SECONDS}s.")
    return False


def do_work():
    """Placeholder for the real task. For the dummy-instance test this just
    holds the instance up for RUN_SECONDS so you can watch the full
    start -> running -> stop cycle. Swap this out for the real handwritten
    OCR call once you're happy with the timing."""
    print(f"[work] simulating {RUN_SECONDS}s of work on the instance...")
    remaining = RUN_SECONDS
    while remaining > 0:
        step = min(10, remaining)
        print(f"[work] {remaining}s left...")
        time.sleep(step)
        remaining -= step
    print("[work] done.")


def main():
    if "PASTE_YOUR" in AWS_ACCESS_KEY_ID or "PASTE_YOUR" in AWS_SECRET_ACCESS_KEY:
        sys.exit("Fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY at the top of this file first.")

    client = get_client()
    start_instance(client)

    if HEALTH_URL:
        if not wait_for_endpoint_ready():
            print("[ec2] health check failed, stopping instance and exiting.")
            stop_instance(client)
            sys.exit(1)

    try:
        do_work()
    finally:
        # guaranteed stop even if do_work() raises or you Ctrl+C mid-run
        stop_instance(client)


if __name__ == "__main__":
    main()