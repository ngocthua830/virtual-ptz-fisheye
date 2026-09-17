import sys, glob, os, statistics as st
sys.path.insert(0,'/app')
import numpy as np
from evaluation.loaf import load_annotations, sequences
ann=load_annotations('/loaf','val')
print(f"{'seq':>6} {'frames':>7} {'annotated':>10} {'empty':>7} {'stride':>7} {'ppl/frm var':>12}")
for seq in sequences('/loaf','val'):
    files=sorted(glob.glob(f'/loaf/images/val/{seq}_*.jpg'))[:120]
    nums=[int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    stride=int(st.median(np.diff(nums))) if len(nums)>1 else 0
    counts=[len(ann.get(seq,{}).get(os.path.basename(f),[])) for f in files]
    empty=sum(1 for c in counts if c==0)
    print(f"{seq:>6} {len(files):>7} {sum(1 for c in counts if c>0):>10} {empty:>7} {stride:>7} {np.std(counts):>12.1f}")
