# try_regions.py  -- run from C:\ocr:  python -m try_regions
import re, requests, boto3
from botocore.config import Config
from sqlalchemy.engine import make_url
import config
from data import data_fetcher as df

# 1) region hint from the DB host (make_url copes with '#' in the password)
host = make_url(config.DATABASE_URL).host or ""
m = re.search(r"aws-\d+-([a-z0-9-]+)\.pooler\.supabase\.com", host)
print("DB host:", host, "| region hint:", m.group(1) if m else "none (direct host)")

ref = re.search(r"https://([a-z0-9]+)\.supabase\.co", df.SUPABASE_URL).group(1)
key = df._list_database_images()[0]["storage_key"]

regions = ([m.group(1)] if m else []) + [
    "ap-south-1", "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ap-northeast-2",
    "us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-west-2",
    "eu-west-3", "eu-central-1", "eu-north-1", "sa-east-1", "ca-central-1"]
regions = list(dict.fromkeys(regions))
endpoints = [f"https://{ref}.supabase.co/storage/v1/s3",
             f"https://{ref}.storage.supabase.co/storage/v1/s3"]

found = None
for ep in endpoints:
    for rg in regions:
        c = boto3.client("s3", endpoint_url=ep, region_name=rg,
                         aws_access_key_id=df._S3_ACCESS_KEY_ID,
                         aws_secret_access_key=df._S3_SECRET_ACCESS_KEY,
                         config=Config(s3={"addressing_style": "path"}))
        url = c.generate_presigned_url("get_object",
              Params={"Bucket": df.SUPABASE_BUCKET, "Key": key}, ExpiresIn=300)
        r = requests.get(url, timeout=30)
        code = re.search(r"<Code>([^<]+)</Code>", r.text)
        print(f"{r.status_code}  {rg:15} {ep.split('//')[1].split('/')[0]:45} {code.group(1) if code else ''}")
        if r.status_code == 200 and not found:
            found = (ep, rg)
print("\nWORKS WITH:", found if found else "nothing -> the key/secret pair itself is likely wrong")