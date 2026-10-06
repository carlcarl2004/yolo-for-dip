#!/bin/bash
# Collect every result artifact into one tarball for the local repo.
set -e
P=/root/autodl-tmp
OUT=$P/export_bundle
rm -rf $OUT; mkdir -p $OUT/eval_all $OUT/night $OUT/runs
cp $P/eval_all/*.json $OUT/eval_all/ 2>/dev/null || true
cp $P/eval_all/*.jpg  $OUT/eval_all/ 2>/dev/null || true
cp $P/night/*.log $P/night/*.json $OUT/night/ 2>/dev/null || true
cp $P/audit/audit.json $OUT/ 2>/dev/null || true
for r in n_merged s_merged m_merged s_rot s_960m n_960m; do
  if [ -d $P/trash_project/runs/$r ]; then
    mkdir -p $OUT/runs/$r
    cp $P/trash_project/runs/$r/results.csv $OUT/runs/$r/ 2>/dev/null || true
    cp $P/trash_project/runs/$r/*.png $OUT/runs/$r/ 2>/dev/null || true
    cp $P/trash_project/runs/$r/args.yaml $OUT/runs/$r/ 2>/dev/null || true
  fi
done
cp $P/datasets/trash_merged_bin/data.yaml $OUT/ 2>/dev/null || true
tar czf $P/export_bundle.tar.gz -C $P export_bundle
ls -la $P/export_bundle.tar.gz