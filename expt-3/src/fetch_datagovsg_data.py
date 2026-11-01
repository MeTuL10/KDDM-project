"""
Robust sequential downloader for Singapore HDB Rental Transactions dataset
from data.gov.sg (Dataset ID: d_c9f57187485a850908655db0e8cfe651)
with rate-limit handling and progress persistence.
"""

import os
import json
import time
import urllib.request
import pandas as pd

RESOURCE_ID = "d_c9f57187485a850908655db0e8cfe651"
BASE_URL = "https://data.gov.sg/api/action/datastore_search"
BATCH_SIZE = 10000
TOTAL_RECORDS = 209852
CACHE_FILE = "expt-3/data/datagovsg_raw_chunks.jsonl"
FINAL_FILE = "expt-3/data/hdb_rental_records_datagovsg.parquet"

def fetch_chunk(offset, limit=BATCH_SIZE, max_retries=5):
    url = f"{BASE_URL}?resource_id={RESOURCE_ID}&limit={limit}&offset={offset}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["result"]["records"]
        except Exception as e:
            wait_sec = 2 ** (attempt + 1)
            print(f"[Warn] Offset {offset} attempt {attempt+1} failed ({e}). Backing off for {wait_sec}s...")
            time.sleep(wait_sec)
    raise RuntimeError(f"Failed to fetch chunk at offset {offset} after {max_retries} attempts.")

def main():
    os.makedirs("expt-3/data", exist_ok=True)
    
    # Check already downloaded offsets
    downloaded_offsets = set()
    total_downloaded = 0
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    downloaded_offsets.add(item["offset"])
                    total_downloaded += len(item["records"])
        print(f"Resuming: found {len(downloaded_offsets)} cached chunks ({total_downloaded} records).")
    
    all_offsets = list(range(0, TOTAL_RECORDS + BATCH_SIZE, BATCH_SIZE))
    
    with open(CACHE_FILE, "a") as f_out:
        for offset in all_offsets:
            if offset in downloaded_offsets:
                continue
            
            print(f"Fetching offset {offset} / {TOTAL_RECORDS}...")
            records = fetch_chunk(offset)
            if not records:
                print(f"No more records at offset {offset}.")
                break
                
            f_out.write(json.dumps({"offset": offset, "records": records}) + "\n")
            f_out.flush()
            total_downloaded += len(records)
            print(f"  -> Got {len(records)} records (Total: {total_downloaded}).")
            time.sleep(1.2) # Gentle pacing to avoid 429
            
    print(f"\nAll chunks downloaded. Assembling full parquet dataset...")
    all_rows = []
    with open(CACHE_FILE, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                all_rows.extend(item["records"])
                
    df = pd.DataFrame(all_rows)
    print(f"Raw dataframe shape: {df.shape}")
    
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    df = df.drop_duplicates()
    df["monthly_rent"] = pd.to_numeric(df["monthly_rent"], errors="coerce")
    
    df.to_parquet(FINAL_FILE, index=False)
    print(f"Successfully saved clean dataset to {FINAL_FILE} (Shape: {df.shape})")
    print(f"Rent range: {df['monthly_rent'].min()} to {df['monthly_rent'].max()}")
    print(f"Date range: {df['rent_approval_date'].min()} to {df['rent_approval_date'].max()}")

if __name__ == "__main__":
    main()
