import sys, time
sys.path.insert(0, '.')
import torch
from qrs_ensemble import QRSEnsemble
from qrs_ensemble_deferred import DeferredSyncQRSEnsemble
from qrs_ensemble_grouped import GroupedQRSEnsemble

torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0), torch.__version__)

ens = QRSEnsemble.load('megamodel_loo32.pt', device='cuda')
deferred = DeferredSyncQRSEnsemble.from_ensemble(ens)
grouped = GroupedQRSEnsemble.from_ensemble(ens)

def bench(fn, B, N=10):
    window = torch.randn(B, 12, 2, 550, device='cuda')
    emb = torch.randn(B, 12, 38, 768, device='cuda')
    for _ in range(3):
        fn(window, emb)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        fn(window, emb)
    torch.cuda.synchronize()
    return (time.time() - t0) / N

print(f"{'B':>5} {'deferred_ms':>12} {'deferred_us/beat':>17} {'grouped_ms':>12} {'grouped_us/beat':>16} {'grouped_speedup':>16}")
for B in (8, 32, 64, 128, 256, 512):
    t_def = bench(deferred.predict, B)
    t_grp = bench(grouped.predict, B)
    print(f"{B:>5} {t_def*1000:>12.1f} {t_def*1e6/B:>17.1f} {t_grp*1000:>12.1f} {t_grp*1e6/B:>16.1f} {t_def/t_grp:>15.2f}x")
