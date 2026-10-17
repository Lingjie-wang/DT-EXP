"""Download the public AuctionNet final/general data without changing its rows."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/"
GROUPS = ("7-8", "14-15", "9-10", "11-12", "13", "16-17", "18-19", "20-21")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download_group(root, group):
    name = f"autoBidding_general_track_final_data_period_{group}.zip"
    archive = root / "archives" / name
    url = BASE + name
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as r:
        expected_size = int(r.headers["Content-Length"])
        etag = r.headers.get("ETag")
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Oversized partial download: {partial}")
        if offset < expected_size:
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            print(f"DOWNLOAD {group} {offset}/{expected_size}", flush=True)
            with urllib.request.urlopen(request, timeout=120) as source:
                if offset and source.status != 206:
                    raise ValueError("Server did not honor download resume")
                with partial.open("ab") as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        if partial.stat().st_size != expected_size:
            raise ValueError(f"Incomplete download: {partial}")
        partial.rename(archive)
    if archive.stat().st_size != expected_size:
        raise ValueError(f"Existing archive has unexpected size: {archive}")
    record = {"url": url, "bytes": expected_size, "etag": etag,
              "sha256": digest(archive), "files": []}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = member.filename
            if ("/" in name or not name.startswith("period-")
                    or not name.endswith(".csv")):
                continue
            period = int(name.removeprefix("period-").removesuffix(".csv"))
            if not 7 <= period <= 20:
                continue
            target = root / "raw" / name
            if not target.exists():
                partial = target.with_suffix(".csv.partial")
                print(f"EXTRACT {name} {member.file_size}", flush=True)
                # Reading through EOF validates the ZIP member's CRC.
                with source.open(member) as reader, partial.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
                if partial.stat().st_size != member.file_size:
                    raise ValueError(f"Incomplete extraction: {name}")
                partial.rename(target)
            if target.stat().st_size != member.file_size:
                raise ValueError(f"Existing CSV has unexpected size: {target}")
            record["files"].append({"name": name, "bytes": member.file_size,
                                    "zip_crc": member.CRC, "sha256": digest(target)})
    (root / "archives" / f"{group}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"READY {group}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    for name in ("archives", "raw"):
        (args.root / name).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_group, args.root, group)
                   for group in args.groups]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
