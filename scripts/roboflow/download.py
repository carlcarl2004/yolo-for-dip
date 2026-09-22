import os, sys, json
from roboflow import Roboflow

KEY = "W3RJsgNxSfcMj5WW2Vq6"
DEST = r"D:\Y3\dip\rf_trash"
os.makedirs(DEST, exist_ok=True)
rf = Roboflow(api_key=KEY)
proj = rf.workspace("technological-institute-of-the-philippines").project("yolov7-trash-05-04-2023")
print("project:", proj.name)
ver = proj.version(1)
print("downloading v1 as yolov7 ...")
ds = ver.download("yolov7", location=DEST)
print("location:", ds.location)
for root, dirs, files in os.walk(ds.location):
    lvl = root.replace(ds.location, "").count(os.sep)
    if lvl > 2: continue
    print("  " * lvl + os.path.basename(root) + f"/  ({len(files)} files)")
