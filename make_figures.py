"""Rigenera TUTTE le figure del paper nello stile della pubblicazione.

Stile imposto, uniforme per costruzione:
  - tutto in inglese;
  - NESSUN titolo nei grafici: la descrizione sta nella caption;
  - nessun risultato numerico dentro etichette o legende;
  - dimensioni in scala 1:1 con la pagina. La larghezza del testo della classe
    ANS e' circa 150 mm = 5.9 in: le figure sono generate a ~6.3 in e inserite a
    150 mm, quindi la riduzione e' del 5% e i caratteri restano leggibili.
    (Prima erano generate a 10.5 in e ridotte al 56%: da qui i grafici piccoli.)

Legge i risultati salvati in ../results/; l'unica cosa ricalcolata sono le
coppie a distanza uno e una singola tribu' con la popolazione registrata.
"""
import sys, os, json, pathlib, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
os.environ.setdefault('LFR_MODEL_DIR', os.path.abspath('../final_model_hier_v2'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

RES = pathlib.Path('../results')
FIG = pathlib.Path('figs')
FIG.mkdir(exist_ok=True)
W1 = 6.3                                   # larghezza standard, pollici

FAM = {'beta': '#8E5BB5', 'flux': '#B5502D', 'keff': '#2F7D5A',
       'life': '#B98419', 'powr': '#3B7A99'}
PRED, TRUE, OK, GREY = '#3B7A99', '#B5502D', '#2F7D5A', '#8A9BA3'
FAMS = ['beta', 'flux', 'keff', 'life', 'powr']
CFGS = ['CRout', 'CRin0', 'CRin55']
CFGLBL = {'CRout': 'withdrawn', 'CRin0': 'partially inserted',
          'CRin55': 'fully inserted'}
OUTS = ['beta_CRin0', 'beta_CRin55', 'beta_CRout', 'flux_CRin0', 'flux_CRin55',
        'flux_CRout', 'keff_CRin0', 'keff_CRin55', 'keff_CRout', 'life_CRin0',
        'life_CRin55', 'life_CRout', 'powr_CRin0', 'powr_CRin55', 'powr_CRout']
SHORT = {'CRout': 'withdr.', 'CRin0': 'partial', 'CRin55': 'inserted'}
LBL = {c: f"{c.split('_')[0]} ({SHORT[c.split('_')[1]]})" for c in OUTS}
EN = np.array([2.0E+01,3.6787940E+00,2.2313020E+00,1.3533530E+00,8.2085E-01,4.9787070E-01,
    3.0197380E-01,1.8315640E-01,1.1109E-01,6.7379470E-02,4.0867710E-02,2.4787520E-02,
    1.5034390E-02,9.1188200E-03,5.5308440E-03,3.3546260E-03,2.0346840E-03,1.2340980E-03,
    7.4851830E-04,4.5399930E-04,3.0432480E-04,1.4862540E-04,9.1660880E-05,6.7904050E-05,
    4.0169E-05,2.2603290E-05,1.3709590E-05,8.3152870E-06,4.0E-06,5.4E-07,1.00001E-11])

plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 8,
    'axes.labelsize': 8, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.3,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})
save = lambda n: (plt.savefig(FIG / n, bbox_inches='tight'), plt.close())


# ══ Fig. — predicted against reference ═══════════════════════════════════
# 5 righe (famiglie) x 3 colonne (configurazioni): pannelli larghi il doppio
# rispetto alla disposizione 3x5, a parita' di larghezza di pagina.
d = json.load(open(RES / 'refined_plot_data.json'))
Yt, Yp, Gv = np.array(d['true']), np.array(d['pred_new']), np.array(d['G'])
Ytr, Ypr = Yt.copy(), Yp.copy()
for i, c in enumerate(OUTS):
    if not c.startswith('powr'):
        Ytr[:, i] = np.log10(Yt[:, i] + 1e-6)
        Ypr[:, i] = np.log10(np.maximum(Yp[:, i], 1e-12) + 1e-6)

fig, ax = plt.subplots(5, 3, figsize=(W1, 8.6), constrained_layout=True)
sc = None
for fi, fam in enumerate(FAMS):
    for ci, cfg in enumerate(CFGS):
        i = OUTS.index(f'{fam}_{cfg}')
        a = ax[fi, ci]
        sc = a.scatter(Ytr[:, i], Ypr[:, i], c=Gv, cmap='viridis', s=3.5,
                       alpha=.6, linewidths=0, vmin=Gv.min(), vmax=Gv.max())
        lo = min(Ytr[:, i].min(), Ypr[:, i].min())
        hi = max(Ytr[:, i].max(), Ypr[:, i].max())
        pad = .05 * (hi - lo)
        a.plot([lo, hi], [lo, hi], '--', color='#444', lw=.8, zorder=0)
        a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad)
        if fi == 0:
            a.set_title(CFGLBL[cfg], fontsize=8.5, pad=4)
        if fi == 4:
            a.set_xlabel('reference')
        if ci == 0:
            a.set_ylabel(f'{fam}\npredicted')
cb = fig.colorbar(sc, ax=ax, shrink=.4, pad=.015, aspect=30)
cb.set_label('number of energy groups')
save('fig_scatter.pdf')

# ══ Fig. — learning curve (campagna adattiva, ensemble da 5) ═════════════
lc = json.load(open(RES / 'al_replay_5nets.json'))['uncertainty']
n = [r['n'] for r in lc]
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.5), constrained_layout=True)
ax[0].plot(n, [r['median_rel_err_pct'] for r in lc], 'o-', color=TRUE, ms=4)
ax[0].set_ylabel('median relative error [%]')
ax[1].plot(n, [r['r2_ext_mean'] for r in lc], 'o-', color=OK, ms=4)
ax[1].set_ylabel('mean coefficient of determination')
for a in ax:
    a.set_xlabel('configurations in the training set')
    a.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
save('fig_learning.pdf')

# ══ Fig. — steepness of the target ═══════════════════════════════════════
import retrain_hier as R
Xb, Yb = R._load_pairs(R.DATA_FILES, R.DATA_DIR)
ok = ~(Yb >= 1e11).any(axis=1)
Ytb = R._to_proc(Yb[ok])
B = (Xb[ok][:, :29] > 0.5)
idx = {b.tobytes(): i for i, b in enumerate(B)}
pairs = []
for i, b in enumerate(B):
    for j in range(29):
        b2 = b.copy(); b2[j] = ~b2[j]
        k = idx.get(b2.tobytes())
        if k is not None and k > i:
            pairs.append((i, k))
pairs = np.array(pairs)
diag = json.load(open(RES / 'diagnose_small.json'))

fig, ax = plt.subplots(1, 3, figsize=(W1, 2.3), constrained_layout=True)
for fam in FAMS:
    col = f'{fam}_CRin0'; i = OUTS.index(col)
    a_, b_ = Ytb[pairs[:, 0], i], Ytb[pairs[:, 1], i]
    mid, jump = (a_ + b_) / 2, np.abs(a_ - b_)
    q = np.quantile(mid, np.linspace(0, 1, 9))
    xs, ys = [], []
    for k in range(len(q) - 1):
        m = (mid >= q[k]) & (mid <= q[k + 1] if k == len(q) - 2 else mid < q[k + 1])
        if m.sum() < 50:
            continue
        xs.append(10 ** ((q[k] + q[k + 1]) / 2) - 1e-6); ys.append(np.median(jump[m]))
    ax[0].plot(xs, ys, 'o-', color=FAM[fam], label=fam, ms=3.2)
    rows = diag[col]
    xr = [10 ** ((r['lo'] + r['hi']) / 2) - 1e-6 if fam != 'powr'
          else (r['lo'] + r['hi']) / 2 for r in rows]
    ax[1].plot(xr, [r['err_rel'] for r in rows], 'o-', color=FAM[fam], ms=3.2)
    sx, sy = [], []
    for r in rows:
        m = (mid >= r['lo']) & (mid < r['hi'])
        if m.sum() < 50:
            continue
        sx.append(np.median(jump[m])); sy.append(r['err_rel'])
    ax[2].scatter(sx, sy, s=18, color=FAM[fam], edgecolor='white', lw=.35)
ax[0].set_xlabel('reference value'); ax[0].set_ylabel('change per boundary\n[decades]')
ax[1].set_xlabel('reference value'); ax[1].set_ylabel('relative error [%]')
ax[2].set_xlabel('change per boundary [decades]'); ax[2].set_ylabel('relative error [%]')
for a in ax:
    a.set_xscale('log'); a.set_yscale('log'); a.grid(alpha=.25, which='both')
ax[0].legend(ncol=2, handlelength=1.2, columnspacing=1.0)
save('fig_steepness.pdf')

# ══ Fig. — behaviour of the searches ═════════════════════════════════════
sols = json.load(open(RES / 'tribes_100_full.json'))
rank = json.load(open(RES / 'verify_all_distinct.json'))['ranking']
ffn = np.array([s['canon_ff_nn'] for s in sols])
Gs = np.array([s['G'] for s in sols])
Bt = np.zeros((len(sols), 29), bool)
for i, s in enumerate(sols):
    for p in s['boundaries']:
        if 2 <= p <= 30:
            Bt[i, p - 2] = True
H = [np.asarray(s['convergenza'], float) for s in sols]
L = min(map(len, H))
q = np.percentile(np.array([h[:L] for h in H]), [10, 50, 90], axis=0)
g = np.arange(1, L + 1)
gen1 = [int(np.argmax(h <= h[-1] * 1.01)) + 1 for h in H]

fig, ax = plt.subplots(1, 3, figsize=(W1, 2.3), constrained_layout=True)
for h in H:
    ax[0].plot(np.arange(1, len(h) + 1), h, color=GREY, lw=.3, alpha=.4)
ax[0].fill_between(g, q[0], q[2], color=PRED, alpha=.22)
ax[0].plot(g, q[1], color=PRED, lw=1.6)
ax[0].set_yscale('log'); ax[0].set_xlabel('generation')
ax[0].set_ylabel('best fitness of the tribe')
ax[1].hist(gen1, bins=np.arange(0, 105, 8), color=OK, alpha=.85, edgecolor='white', lw=.4)
ax[1].set_xlabel('generation of convergence'); ax[1].set_ylabel('number of tribes')
ax[2].scatter(Gs, ffn, s=14, color=PRED, alpha=.5, edgecolor='white', lw=.25,
              label='metamodel')
for r in rank:
    ax[2].annotate('', xy=(r['G'], float(r['canon_ff_true'])),
                   xytext=(r['G'], float(r['canon_ff_nn'])),
                   arrowprops=dict(arrowstyle='->', color=TRUE, lw=.7, alpha=.85))
ax[2].scatter([r['G'] for r in rank], [float(r['canon_ff_true']) for r in rank],
              s=18, color=TRUE, marker='D', zorder=3, label='reference solver')
ax[2].set_yscale('log'); ax[2].set_xlabel('number of energy groups')
ax[2].set_ylabel('joint fitness')
ax[2].legend(handlelength=1.2, loc='upper right')
save('fig_search.pdf')

# ══ Fig. — structure of the solutions ════════════════════════════════════
uniq, seen = [], set()
for i, s in enumerate(sols):
    k = tuple(s['boundaries'])
    if k not in seen:
        seen.add(k); uniq.append(i)
Bu, ffu = Bt[uniq], ffn[uniq]
D = (Bu[:, None, :] != Bu[None, :, :]).sum(-1).astype(float)
emb = MDS(n_components=2, dissimilarity='precomputed', random_state=0,
          normalized_stress='auto').fit_transform(D)
best = min(sols, key=lambda s: s['canon_ff_nn'])
top = ffn <= np.percentile(ffn, 40)
Ei = EN[1:30]

fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6), constrained_layout=True,
                       gridspec_kw={'width_ratios': [1.55, 1]})
ax[0].bar(Ei, Bt[top].mean(0) * 100, width=Ei * .42, color=OK, alpha=.85,
          label='better tribes')
ax[0].bar(Ei, -Bt[~top].mean(0) * 100, width=Ei * .42, color=GREY, alpha=.75,
          label='remaining tribes')
for p in best['boundaries']:
    if 2 <= p <= 30:
        ax[0].axvline(EN[p - 1], color=TRUE, lw=.6, ls=':', alpha=.85)
ax[0].axhline(0, color='#333', lw=.6)
ax[0].set_xscale('log'); ax[0].invert_xaxis()
ax[0].set_xlabel('energy of the boundary [MeV]'); ax[0].set_ylabel('occurrence [%]')
ax[0].legend(loc='lower left', handlelength=1.2)
ax[0].grid(alpha=.25, axis='x', which='both')
s2 = ax[1].scatter(emb[:, 0], emb[:, 1], c=np.log10(ffu), s=26, cmap='viridis_r',
                   edgecolor='white', lw=.35, zorder=3)
ax[1].scatter(emb[int(np.argmin(ffu)), 0], emb[int(np.argmin(ffu)), 1], s=110,
              facecolor='none', edgecolor=TRUE, lw=1.2, zorder=4)
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].grid(False)
ax[1].set_xlabel('first MDS coordinate'); ax[1].set_ylabel('second MDS coordinate')
cb = plt.colorbar(s2, ax=ax[1]); cb.set_label('joint fitness (log scale)')
save('fig_structure.pdf')

# ══ Fig. — optimism of the surrogate ═════════════════════════════════════
tv = np.array([float(r['canon_ff_true']) for r in rank])
pv = np.array([float(r['canon_ff_nn']) for r in rank])
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6), constrained_layout=True)
lim = [min(tv.min(), pv.min()) * .94, max(tv.max(), pv.max()) * 1.06]
ax[0].fill_between(lim, lim, [lim[0], lim[0]], color=TRUE, alpha=.07)
ax[0].plot(lim, lim, '--', color='#444', lw=.8, zorder=1)
sc = ax[0].scatter(tv, pv, s=26, c=[r['G'] for r in rank], cmap='viridis',
                   edgecolor='white', lw=.35, zorder=3)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('fitness from the reference solver')
ax[0].set_ylabel('fitness from the metamodel')
from matplotlib.ticker import LogLocator, NullFormatter
for a in (ax[0].xaxis, ax[0].yaxis):
    a.set_major_locator(LogLocator(base=10, numticks=4))
    a.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=12))
    a.set_minor_formatter(NullFormatter())
cb = plt.colorbar(sc, ax=ax[0]); cb.set_label('energy groups')
pos = np.arange(15)
bp = ax[1].boxplot([[r['rels'][c] for r in rank] for c in OUTS], positions=pos,
                   widths=.6, patch_artist=True, showfliers=False)
for b, c in zip(bp['boxes'], OUTS):
    b.set(facecolor=FAM[c.split('_')[0]], alpha=.75, edgecolor='#444', lw=.5)
for m in bp['medians']:
    m.set(color='#222', lw=1.0)
for w in bp['whiskers'] + bp['caps']:
    w.set(lw=.5)
ax[1].set_yscale('log'); ax[1].set_xticks(pos)
ax[1].set_xticklabels([LBL[c] for c in OUTS], rotation=68, ha='right', fontsize=6)
ax[1].set_ylabel('relative error of the metamodel [%]')
ax[1].grid(alpha=.25, axis='y', which='both')
save('fig_bias.pdf')

# ══ Fig. — accuracy against number of groups ═════════════════════════════
tg = json.load(open(RES / 'tradeoff_G.json'))
sd = json.load(open(RES / 'baseline_standard.json'))
okp = [c for c in tg if c['ff_true']]
fig, ax = plt.subplots(figsize=(4.4, 2.9), constrained_layout=True)
ax.plot([c['G'] for c in okp], [c['ff_true'] for c in okp], 'o-', color=OK,
        ms=4.5, label='reference solver')
ax.plot([c['G'] for c in tg], [c['ff_nn'] for c in tg], 's--', color=GREY,
        ms=3, lw=.9, label='metamodel')
for r in sd['standard']:
    ax.plot([r['G']], [r['ff']], '*', ms=11, color=TRUE, zorder=4)
    ax.annotate('equal lethargy' if 'letargia' in r['nome'] else 'uniform index',
                (r['G'], r['ff']), fontsize=7, color=TRUE,
                xytext=(5, 3), textcoords='offset points')
r29 = min(json.load(open(RES / 'ref29.json')), key=lambda r: r['canon_ff_true'])
ax.plot([29], [r29['canon_ff_true']], '*', ms=11, color=TRUE, zorder=4)
ax.annotate('finest admissible', (29, r29['canon_ff_true']), fontsize=7, color=TRUE,
            ha='right', xytext=(-4, 2), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('number of energy groups')
ax.set_ylabel('best attainable joint fitness')
ax.grid(alpha=.25, which='both'); ax.legend(loc='center right', handlelength=1.4)
save('fig_tradeoff.pdf')

# ══ Fig. — evolution of a single tribe ═══════════════════════════════════
# La mappa occupa tutta la larghezza; la fitness sta sotto, non a fianco.
import lfr_ga as G_
import lfr_workflow as W_
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA as _GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class PopCapture(Callback):
    def __init__(self):
        super().__init__()
        self.pop, self.fit = [], []

    def notify(self, algorithm):
        X = algorithm.pop.get('X')
        self.pop.append(np.array([G_.decode_chromosome(x) for x in X], dtype=bool))
        self.fit.append(np.asarray(algorithm.pop.get('F')).ravel().copy())


win = W_.canonical_window()
np.random.seed(1000)
prob = W_._make_canonical_problem(win)(G_.nn_model, G_.scaler_X, G_.scaler_Y,
                                       G_.ACTIVE_CONFIGS, G_.ACTIVE_FAMILIES)
cbk = PopCapture()
res = minimize(prob, _GA(pop_size=100, sampling=G_._sampling, crossover=G_._crossover,
                         mutation=G_._mutation, repair=G_.ChromosomeRepair(),
                         eliminate_duplicates=True),
               get_termination('n_gen', 100), callback=cbk, seed=1000, verbose=False)
P = np.array([p.mean(0) for p in cbk.pop])
Fq = np.array([np.percentile(f, [10, 50, 90]) for f in cbk.fit])
Fb = np.array([f.min() for f in cbk.fit])
final = G_.decode_chromosome(res.X.ravel()).astype(bool)
gen = np.arange(1, len(P) + 1)

fig, ax = plt.subplots(2, 1, figsize=(W1, 4.6), constrained_layout=True,
                       gridspec_kw={'height_ratios': [1.9, 1]}, sharex=True)
im = ax[0].imshow(P.T, aspect='auto', origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1,
                  extent=[1, len(P), -0.5, 28.5], interpolation='nearest')
for j in np.flatnonzero(final):
    ax[0].plot([1, len(P)], [j, j], color='white', lw=.5, ls=(0, (2.5, 2.5)), alpha=.9)
tk = [0, 4, 9, 14, 19, 24, 28]
ax[0].set_yticks(tk); ax[0].set_yticklabels([f'{Ei[t]:.3g}' for t in tk])
ax[0].invert_yaxis()
ax[0].set_ylabel('energy of the boundary [MeV]')
ax[0].grid(False)
cb2 = fig.colorbar(im, ax=ax[0], pad=.012, aspect=18)
cb2.set_label('fraction of the population\ncontaining the boundary')
ax[1].fill_between(gen, Fq[:, 0], Fq[:, 2], color=PRED, alpha=.22,
                   label='tenth to ninetieth percentile')
ax[1].plot(gen, Fq[:, 1], color=PRED, lw=1.3, label='median of the population')
ax[1].plot(gen, Fb, color=TRUE, lw=1.5, label='best individual')
ax[1].set_yscale('log'); ax[1].set_xlabel('generation')
ax[1].set_ylabel('joint fitness')
ax[1].legend(loc='upper right', handlelength=1.4)
ax[1].set_xlim(1, len(P))
save('fig_tribe.pdf')

# (il confronto con la selezione casuale resta in results/, fuori dal paper)


# ══ Fig. — recurring boundaries ══════════════════════════════════════════
# A sinistra la ricorrenza confine per confine con il peso della fitness sullo
# sfondo; a destra le due letture in competizione, il peso e la forma.
bp = json.load(open(RES / 'boundary_patterns.json'))
Eb = np.array(bp['energie_MeV']); rec = np.array(bp['ricorrenza']) * 100
wg = np.array(bp['importanza_gruppo']); kk = np.array(bp['kink_spettro'])
wb = .5 * (wg[:-1] + wg[1:]) * 100

fig = plt.figure(figsize=(W1, 4.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
a0 = fig.add_subplot(gs[0, :])
a0.bar(Eb, rec, width=Eb * .42, color=OK, alpha=.85, zorder=3)
a0.set_xscale('log'); a0.invert_xaxis(); a0.set_ylim(0, 105)
a0.set_xlabel('energy of the boundary [MeV]')
a0.set_ylabel('occurrence among\nthe solutions [%]')
a0.grid(alpha=.25, axis='x', which='both')
a1 = a0.twinx()
a1.plot(Eb, wb, color=PRED, lw=1.1, marker='o', ms=2.6, zorder=4)
a1.set_ylabel('total importance [%]', color=PRED)
a1.tick_params(axis='y', colors=PRED); a1.grid(False)
a1.spines['right'].set_visible(True); a1.spines['right'].set_color(PRED)
for k, (x, lab) in enumerate(((wb, 'total importance around the boundary [%]'),
                              (kk, 'variation of the flux shape across it'))):
    a = fig.add_subplot(gs[1, k])
    sc = a.scatter(x, rec, s=24, c=np.log10(Eb), cmap='viridis_r',
                   edgecolor='white', lw=.3, zorder=3)
    a.set_xlabel(lab); a.set_ylim(-5, 105)
    a.set_ylabel('occurrence [%]' if k == 0 else '')
    if k: a.set_yticklabels([])
cb = fig.colorbar(sc, ax=fig.axes[-1], pad=.03)
cb.set_label('energy of the boundary [MeV, log scale]')
cb.set_ticks([])
save('fig_boundaries.pdf')

print(f'{len(list(FIG.glob("fig_*.pdf")))} figure rigenerate, larghezza {W1} in '
      'per stampa a 150 mm')
for f in sorted(FIG.glob('fig_*.pdf')):
    print(f'  {f.name:<22} {f.stat().st_size/1024:6.0f} kB')
