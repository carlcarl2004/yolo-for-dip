#!/usr/bin/env python
"""多线程下载 TACO 数据集图片。

  python download_taco.py --out data/images --workers 32            # 只下 640px 小图（快，推荐）
  python download_taco.py --out data/images --workers 32 --full     # 连原图一起下（S3 源很慢）
"""
import argparse
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

AP = "/root/autodl-tmp/TACO/data/annotations.json"
done = 0
fail = []


def pick_url(img, full):
    if full:
        return img.get("flickr_640_url") or img["flickr_url"]
    return img.get("flickr_640_url")


def fetch(img, out_root, full, retries=3):
    url = pick_url(img, full)
    if not url:
        return "skip"  # 没有小图源，跳过
    dst = os.path.join(out_root, img["file_name"])
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r, open(dst + ".part", "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dst + ".part", dst)
            return "ok"
        except Exception as e:
            if attempt == retries - 1:
                fail.append((img["file_name"], str(e)[:80]))
                return "fail"
    return "fail"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", default=AP)
    p.add_argument("--out", default="/root/autodl-tmp/TACO/data/images")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--full", action="store_true", help="连没有小图源的原图一起下（很慢）")
    a = p.parse_args()

    imgs = json.load(open(a.ann))["images"]
    if not a.full:
        imgs = [i for i in imgs if i.get("flickr_640_url")]
        print(f"只下 640px 小图：{len(imgs)} 张（跳过大图源）")
    else:
        print(f"下载全部 {len(imgs)} 张（含大图，可能很慢）")

    results = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, im, a.out, a.full): im for im in imgs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r] += 1
            if i % 100 == 0 or i == len(imgs):
                print(f"[{i}/{len(imgs)}] ok={results['ok']} skip={results['skip']} fail={results['fail']}", flush=True)

    n = sum(len(fs) for _, _, fs in os.walk(a.out))
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"完成：{results}，磁盘上 {n} 个文件 / {sz/1e6:.1f} MB")
    if fail:
        with open("/root/autodl-tmp/TACO/failed_urls.txt", "w") as f:
            for name, err in fail:
                f.write(f"{name}\t{err}\n")
        print(f"失败 {len(fail)} 个，已写入 failed_urls.txt")


if __name__ == "__main__":
    main()
