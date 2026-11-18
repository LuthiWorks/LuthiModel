"""Can the memory search be exact and fast? A trial on made-up stores.

Fable 5.2 (chat), 2026-11-18, for build brief 10. Terminal Fable measured the
exact search (8-byte arithmetic) at 0.19 to 0.28 s after one day of memories,
and one 4-byte product at 0.037 s. A 4-byte search can bring back a different
memory. This tries a third way: one 4-byte pass gives every row an upper and
a lower bound on its likeness, and only the rows that could be the best are
computed exactly. The answer is then the exact search's answer, by
arithmetic and not by luck, as long as the bounds hold.

    python redteam/fable_2026_11_18/probe_screened_search.py 100000 4

(rows, threads). Three made-up stores at 768 wide. On the cloud box, 100,000
rows, 4 threads: exact 129 to 146 ms, screened 18 to 23 ms, 0 of 90 answers
differ, at most 487 rows recomputed. Made-up stores only: none of them makes
the plain 4-byte search differ, so they are not hard cases. The real test is
the archived stores and a hard made-up one (brief 10).

The bounds: a 4-byte dot of d terms is within (d + 2) * 2^-24 * |r| * |c| of
the true one (the classical bound, and the cue's rounding). A row's distance
from the mean is kept exactly against a reference mean, and the mean's drift
since then widens the bound.
"""
import torch, time, sys
torch.manual_seed(1); torch.set_num_threads(int(sys.argv[2]) if len(sys.argv)>2 else 4)
d=768; n=int(sys.argv[1]); U=2.0**-24; CH=1024
def store(kind):
    base=torch.randn(d,dtype=torch.float64)
    if kind=="walk":
        walk=(torch.randn(n,d,dtype=torch.float64)*2e-4).cumsum(0); walk=walk/walk.norm(dim=1).max()*0.045
        return (base/base.norm()+walk).to(torch.float32)
    if kind=="tight":   # everything alike at 0.99999: the worst the archived tiny lives showed
        return (base/base.norm()+torch.randn(n,d,dtype=torch.float64)*1.6e-4).to(torch.float32)
    return torch.randn(n,d,dtype=torch.float64).to(torch.float32)
def blocks(M,f): return torch.cat([f(M[a:a+65536].to(torch.float64)) for a in range(0,M.shape[0],65536)])
def exact(M,mean,cue,idx=None):
    c=cue.to(torch.float64)-mean
    if idx is not None:
        r=M[idx].to(torch.float64)-mean; nr=r.norm(dim=1)*c.norm(); return torch.where(nr>0,(r@c)/nr,torch.zeros_like(nr))
    out=torch.empty(M.shape[0],dtype=torch.float64)
    for a in range(0,M.shape[0],CH):
        r=M[a:a+CH].to(torch.float64)-mean; nr=r.norm(dim=1)*c.norm(); out[a:a+r.shape[0]]=torch.where(nr>0,(r@c)/nr,torch.zeros_like(nr))
    return out
def screen(M,rn,n0,m0,mean,cue):
    """Bounds, never the answer. rn=|r| and n0=|r-m0| per row, kept in 8-byte when a row is filed."""
    c=cue.to(torch.float64)-mean; cn=float(c.norm()); c32=c.to(torch.float32)
    dot=(M@c32).to(torch.float64)                                  # the one pass over the matrix
    err=(d+2)*U*rn*float(c32.to(torch.float64).norm())*1.01        # classical bound on a 4-byte dot of d terms, and the cue's rounding
    num=dot-float(mean@c)
    drift=float((mean-m0).norm())
    lo_den=(n0-drift).clamp_min(0.0); hi_den=n0+drift
    up=torch.where(num>=0,(num+err)/(lo_den*cn).clamp_min(1e-300),(num+err)/(hi_den*cn)).clamp(max=1.0)
    low=torch.where(num>=0,(num-err)/(hi_den*cn),(num-err)/(lo_den*cn).clamp_min(1e-300)).clamp(min=-1.0)
    cand=(up>=low.max()).nonzero().flatten()
    ex=exact(M,mean,cue,cand); j=int(ex.argmax())
    return int(cand[j]), float(ex[j]), int(cand.numel())
for kind in ("walk","tight","random"):
    M=store(kind); m0=blocks(M,lambda b:b.sum(0,keepdim=True)).sum(0)/n
    rn=blocks(M,lambda b:b.norm(dim=1)); n0=blocks(M,lambda b:(b-m0).norm(dim=1))
    # the mean has moved since m0 was taken: as if 2% more rows were filed
    extra=M[torch.randint(n,(n//50,))].to(torch.float64); mean=(m0*n+extra.sum(0))/(n+n//50)
    te=tf=0; differ=0; cands=[]; four=0
    for q in range(30):
        i=int(torch.randint(n,(1,)))
        cue=(M[i].to(torch.float64)+torch.randn(d,dtype=torch.float64)*{"walk":1e-5,"tight":1e-4,"random":0.5}[kind]).to(torch.float32)
        t=time.perf_counter(); e=exact(M,mean,cue); te+=time.perf_counter()-t
        t=time.perf_counter(); b,l,k=screen(M,rn,n0,m0,mean,cue); tf+=time.perf_counter()-t
        differ+= (b!=int(e.argmax())) or (l!=float(e.max())); cands.append(k)
        c32=(cue.to(torch.float64)-mean).to(torch.float32); R=M-mean.to(torch.float32); four+= int(((R@c32)/(R.norm(dim=1)*c32.norm())).argmax())!=int(e.argmax())
    t2=torch.topk(e,2).values
    print(f"{kind:6s} n={n}: exact {te/30*1000:.0f} ms | screened {tf/30*1000:.0f} ms | answers that differ {differ}/30 | plain 4-byte differs {four}/30 | rows recomputed: median {sorted(cands)[15]}, most {max(cands)} | top-2 gap {float(t2[0]-t2[1]):.1e}")
