#!/usr/bin/env python
"""多线程分块下载器（服务器上没有 aria2c 时用）。

  python pget.py <url> <输出文件> [--threads 16] [--chunk-mb 8]

支持断点续传：重复执行会复用已完成的 .partN 分片。
"""
import argparse
import os
import sys
import threading
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def total_size(url: str) -> int:
    request = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, path: str, start: int, end: int, retries: int = 5) -> bool:
    if os.path.exists(path) and os.path.getsize(path) == end - start + 1:
        return True
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                with open(path, "wb") as handle:
                    while True:
                        chunk = response.read(262144)
                        if not chunk:
                            break
                        handle.write(chunk)
            if os.path.getsize(path) == end - start + 1:
                return True
        except Exception as error:
            print(f"  重试 {attempt + 1}/{retries} chunk {start}: {str(error)[:80]}", flush=True)
            time.sleep(3)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args()

    size = total_size(args.url)
    chunk = args.chunk_mb * 1024 * 1024
    parts = [
        (start, min(start + chunk - 1, size - 1))
        for start in range(0, size, chunk)
    ]
    print(f"总大小 {size / 1e6:.1f} MB，分 {len(parts)} 片，{args.threads} 线程", flush=True)

    queue = list(enumerate(parts))
    lock = threading.Lock()
    done = [0]
    failures: list[int] = []

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                index, (start, end) = queue.pop(0)
            ok = fetch_range(args.url, f"{args.output}.part{index}", start, end)
            with lock:
                if ok:
                    done[0] += 1
                    print(f"  完成 {done[0]}/{len(parts)}", flush=True)
                else:
                    failures.append(index)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.threads)]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        print(f"失败分片 {failures}，请重跑本命令续传", flush=True)
        return 1

    with open(args.output, "wb") as out:
        for index in range(len(parts)):
            with open(f"{args.output}.part{index}", "rb") as part:
                while True:
                    block = part.read(1048576)
                    if not block:
                        break
                    out.write(block)
    for index in range(len(parts)):
        os.remove(f"{args.output}.part{index}")
    elapsed = time.time() - started
    print(f"完成 {args.output} {size / 1e6:.1f} MB，用时 {elapsed / 60:.1f} 分钟，"
          f"平均 {size / 1e6 / elapsed:.1f} MB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
