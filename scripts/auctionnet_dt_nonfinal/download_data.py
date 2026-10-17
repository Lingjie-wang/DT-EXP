"""Download the pinned non-final/general data into an independent cache."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

def file_record(path):
    sha, crc = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    return {"sha256": sha.hexdigest(), "crc32": crc}


def download_group(root, entry):
    url, size, etag = entry["url"], entry["bytes"], entry["etag"]
    archive = root / "archives" / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                timeout=60) as response:
        if (int(response.headers["Content-Length"]) != size
                or response.headers.get("ETag") != etag):
            raise ValueError(f"Pinned remote object changed: {url}")
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial: {partial}")
        if offset < size:
            request = urllib.request.Request(url, headers={"If-Match": etag})
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            print(f"DOWNLOAD {entry['group']} {offset}/{size}", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or
                               response.headers.get("Content-Range") !=
                               f"bytes {offset}-{size - 1}/{size}"):
                    raise ValueError("Server did not honor the requested resume range")
                with partial.open("ab") as stream:
                    shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
        if partial.stat().st_size != size:
            raise OSError(f"Incomplete download: {partial}")
        partial.rename(archive)
    if archive.stat().st_size != size:
        raise ValueError(f"Unexpected archive size: {archive}")
    record = {**entry, "sha256": file_record(archive)["sha256"], "files": []}
    periods = [int(p) for p in entry["group"].split("-") if 7 <= int(p) <= 20]
    with zipfile.ZipFile(archive) as source:
        for period in periods:
            name = f"period-{period}.csv"
            member = source.getinfo(name)
            target = root / "raw" / name
            if not target.exists():
                partial = target.with_suffix(".csv.partial")
                print(f"EXTRACT {name} {member.file_size}", flush=True)
                with source.open(member) as reader, partial.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
                partial.rename(target)
            measured = file_record(target)
            if target.stat().st_size != member.file_size or measured["crc32"] != member.CRC:
                raise ValueError(f"Extracted CSV failed size/CRC check: {target}")
            record["files"].append({"name": name, "bytes": member.file_size, **measured})
    (root / "archives" / f"{entry['group']}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(f"READY {entry['group']}", flush=True)
    return record


def download_with_retry(root, entry):
    for attempt in range(5):
        try:
            return download_group(root, entry)
        except (OSError, urllib.error.URLError) as error:
            if attempt == 4:
                raise
            print(f"RETRY {entry['group']} attempt={attempt + 1}: {error}", flush=True)
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("dataset_manifest.json").read_text())
    marker = args.root / "dataset_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != manifest:
        raise ValueError("Cache belongs to a different dataset version")
    for name in ("archives", "raw"):
        (args.root / name).mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda e: download_with_retry(args.root, e),
                                manifest["archives"]))
    (args.root / "download_complete.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
