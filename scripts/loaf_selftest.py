import sys, math, collections, statistics as st
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence, load_annotations, sequences

ROOT='/loaf'
ann = load_annotations(ROOT,'val')
print('sequences:', sequences(ROOT,'val'))
ok=True
for seq in ('0055','0051','0054'):
    s=LoafSequence(ROOT,'val',seq,ann=ann)
    gt=s.reference(max_frames=120)
    per=[len(f) for f in gt]
    life=collections.Counter()
    for f in gt:
        for p in f: life[p['id']]+=1
    L=list(life.values())
    moving=sum(1 for f in gt for p in f if p['moving'])
    tot=sum(per)
    print(f"\n  seq {seq}: {len(gt)} frames, circle=({s.cx:.0f},{s.cy:.0f},r={s.R:.0f})")
    print(f"    people/frame  mean {st.mean(per):.1f}  max {max(per)}")
    print(f"    tracks {len(L)} | lifetime med {st.median(L):.0f} max {max(L)} | singletons {sum(1 for x in L if x==1)}")
    print(f"    moving fraction {moving/max(tot,1):.2f}")
    pol=[p['bearing'][1] for f in gt for p in f]
    print(f"    polar range {min(pol):.1f}..{max(pol):.1f} deg (must be <=90)")
    if max(pol)>90.5: print('    [FAIL] polar beyond horizon'); ok=False
    # a track that lives 1 frame everywhere would mean association is broken
    if st.median(L) < 2: print('    [FAIL] association degenerate'); ok=False
print('\nSELFTEST', 'PASS' if ok else 'FAIL')
