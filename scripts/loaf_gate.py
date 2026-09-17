import sys, statistics as st
sys.path.insert(0,'/app')
from collections import Counter
from evaluation.loaf import LoafSequence, load_annotations, sequences
ann=load_annotations('/loaf','val')
print(f"{'gate':>6} " + " ".join(f"{s:>22}" for s in ('0051','0052','0054')))
print(f"{'(cm)':>6} " + " ".join(f"{'tracks med-life sing%':>22}" for _ in range(3)))
for gate in (150.0, 250.0, 350.0):
    row=[]
    for seq in ('0051','0052','0054'):
        s=LoafSequence('/loaf','val',seq,assoc_gate_cm=gate,ann=ann)
        gt=s.reference(max_frames=120)
        life=Counter()
        for f in gt:
            for p in f: life[p['id']]+=1
        L=list(life.values())
        row.append(f"{len(L):>6} {st.median(L):>8.0f} {100*sum(1 for x in L if x==1)/len(L):>6.0f}")
    print(f"{gate:>6.0f} " + " ".join(f"{r:>22}" for r in row))
print("\nphysical bound: a 5 m/s sprint over the 0.5 s stride = 250 cm")
