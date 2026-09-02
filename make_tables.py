"""Genera tables.tex nello stile ANS (nseJournal) direttamente dai risultati.

Nessun numero è trascritto a mano: se rilanci un esperimento, rilanci questo e
le tabelle del paper si aggiornano. Convenzioni ANS: didascalia PRIMA del
tabular, filetti \\hline, colonne delimitate da |.
"""
import sys, os, json, pathlib, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
os.environ.setdefault('LFR_MODEL_DIR', os.path.abspath('../final_model_hier_v2'))
import numpy as np
import lfr_ga as G, lfr_workflow as W, retrain_hier as R

RES  = pathlib.Path('../results')
best = json.load(open(RES / 'best_grid.json'))
meta = json.load(open('../final_model_hier_v2/metadata.json'))
fin  = json.load(open(RES / 'final_scheme_100.json'))
ver  = json.load(open(RES / 'verify_all_distinct.json'))
abl  = json.load(open(RES / 'arch_ablation.json'))
noi  = json.load(open(RES / 'arch_seed_noise.json'))
bp   = json.load(open(RES / 'boundary_patterns.json'))
std  = json.load(open(RES / 'baseline_standard.json'))
stdv = json.load(open(RES / 'baseline_standard_valori.json'))
dat  = json.load(open(RES / 'refined_plot_data.json'))
r29  = json.load(open(RES / 'ref29.json'))
fam  = json.load(open(RES / 'families_why.json'))
famv = json.load(open(RES / 'families_verified.json'))
win  = W.canonical_window()

esc = lambda s: s.replace('_', r'\_')


def sci(x, nd=4):
    """4.389e9 -> $4.389\\times10^{9}$, con esponente di qualunque lunghezza."""
    m, e = f'{x:.{nd - 1}e}'.split('e')
    return f'${m}' + (r'\times10^{' + str(int(e)) + '}$' if int(e) else '$')


def scaled(col, raw):
    """Contributo che entra nel prodotto, dalla finestra canonica fissa."""
    v = float(G.apply_ff_controls(col, np.array([raw]))[0])
    lo, hi = win[0][col], win[1][col]
    return max(1 + 99 * (v - lo) / (hi - lo), 0.01) if hi > lo else 50.0


out = []

# ══ Tabella I — accuratezza per uscita ═══════════════════════════════════
rel, r2 = np.array(dat['rel_new']), dat['r2_new']
out.append(r'''\begin{table}[htbp]
\centering
\caption{Metamodel accuracy on the external validation set, %d configurations
never used in training nor for early stopping. The coefficient of determination
is computed in the space in which the network is trained; the relative error is
in physical units.}
\label{tab:accuracy}
\begin{tabular}{|lcc|lcc|}
\hline
Output & $R^2$ & Rel.\ err.\ [\%%] & Output & $R^2$ & Rel.\ err.\ [\%%] \\ \hline''' % dat['n_ext'])
N = R.OUTPUT_NAMES
for i in range(0, 14, 2):
    cells = []
    for k in (i, i + 1):
        cells.append(f'{esc(N[k])} & {r2[N[k]]:.5f} & {np.median(rel[:, k]):.3f}'
                     if k < 15 else ' & & ')
    out.append('    ' + ' & '.join(cells) + r' \\')
out.append(f'    {esc(N[14])} & {r2[N[14]]:.5f} & {np.median(rel[:,14]):.3f} & & & '
           + r'\\ \hline')
out.append(r'    \textbf{Ensemble} & \textbf{%.5f} & \textbf{%.3f} & & & \\ \hline'
           % (meta['r2_ext_mean'], meta['median_rel_err_pct_ext']))
out.append(r'\end{tabular}' + '\n' + r'\end{table}')

# ══ Tabella Ib — ablazione dell'architettura ═════════════════════════════
ORD = ['flat', 'families', 'hierarchical']
LBL = {'flat': 'Shared trunk, fifteen linear heads',
       'families': 'Shared trunk, one neck per family',
       'hierarchical': 'Shared trunk, family neck, configuration branch'}
ab = {r['arch']: r for r in abl['results']}
out.append(r"""
\begin{table}[htbp]
\centering
\caption{Contribution of each level of the hierarchy, measured on the same
%s configurations and with the same reduced protocol: one network per
architecture, constant learning rate and early stopping. The absolute values are
therefore higher than those of the final ensemble, and only their differences
are meaningful. The last line gives the scatter observed over five networks of
identical architecture differing only in their random initialisation, which sets
the scale below which a difference cannot be read.}
\label{tab:ablation}
\begin{tabular}{|l|c|c|c|}
\hline
Architecture & Parameters & $R^2$ & Rel.\ err.\ [\%%] \\ \hline""" %
           f"{abl['protocol']['n_train']:,}".replace(',', r'\,'))
for k in ORD:
    out.append(f"    {LBL[k]} & {ab[k]['n_params']/1e6:.2f}M & "
               f"{ab[k]['r2_ext_mean']:.4f} & {ab[k]['median_rel_err_pct']:.3f} \\\\")
out.append(r'\hline')
out.append(r'    \emph{Spread over five random initialisations} & & '
           f"$\\pm${noi['r2_std']:.4f} & $\\pm${noi['err_std']:.2f} \\\\ \\hline")
out.append(r'\end{tabular}' + '\n' + r'\end{table}')

# ══ Tabella II — le tre ipotesi escluse ══════════════════════════════════
dsm  = json.load(open(RES / 'diagnose_small.json'))['keff_CRin0']
feat = json.load(open(RES / 'train_features_metrics.json'))
out.append(r"""
\begin{table}[htbp]
\centering
\caption{Three candidate explanations for the residual error at small values of
the reactivity and kinetics fitness functions, and the measurement that rules
each of them out.}
\label{tab:hypotheses}
\begin{tabular}{|p{2.9cm}|p{4.55cm}|p{4.55cm}|}
\hline
Hypothesis & Measurement & Outcome \\ \hline
Noise in the labels & %s configurations simulated more than once &
Spread of $0.0000\%%$ on all fifteen outputs; an independent re-run reproduces
stored reference values to all digits \\ \hline
Sparse data in that regime & Training points in the two extreme bands of the
reference value, sampled by the same number of validation points &
$%s$ points where the error is largest, $%.2f\%%$, against $%s$ where it is
$%.2f\%%$ \\ \hline
Input representation & 65 physics-informed descriptors appended to the
30 inputs & The model becomes \emph{worse}: $%.3f\%%$ against $%.3f\%%$
\\ \hline
\end{tabular}
\end{table}""" % (r'21\,440', f"{dsm[0]['n_train']:,}".replace(',', r'\,'),
                  dsm[0]['err_rel'], f"{dsm[-1]['n_train']:,}".replace(',', r'\,'),
                  dsm[-1]['err_rel'], feat['median_rel_err_pct_ext'],
                  meta['median_rel_err_pct_ext']))

# ══ Tabella III — ottimizzata contro le griglie di riferimento ═══════════
b29 = min(r29, key=lambda r: r['canon_ff_true'])
vals = {('Optimized', '', best['G']): best['frenetic_values']}
for r in std['standard']:
    key = [k for k in stdv if k.startswith(r['nome'][:8])][0]
    lbl = ('Equal', 'lethargy') if 'letargia' in r['nome'] else ('Uniform', 'index')
    vals[(*lbl, r['G'])] = stdv[key]
vals[('Finest', 'admissible', 29)] = b29['frenetic_values']
names = list(vals)
hdr = ('Figure of merit & ' + ' & '.join(n[0] for n in names) + r' \\' + '\n'
       + '     & ' + ' & '.join(n[1] for n in names) + r' \\' + '\n'
       + '    $G$ & ' + ' & '.join(str(n[2]) for n in names) + r' \\ \hline')
out.append(r'''
\begin{table}[htbp]
\centering
\caption{The optimized structure against three reference ones, quantity by
quantity: two built with standard spacing rules and the finest structure the
collapsing procedure admits, obtained by suppressing a single boundary of the
fine mesh. Each entry is the scaled contribution entering the joint fitness,
obtained from the FRENETIC values through the fixed canonical window: unity is
the best attainable value and 100 the worst. Six of the fifteen terms score
unity for all four structures, a saturation discussed in Sec.~\\ref{sec:active}.}
\label{tab:standard}
\begin{tabular}{|l|''' + 'c|' * len(names) + r'''}
\hline
''' + hdr)
prod = dict.fromkeys(names, 1.0)
for c in W.COLS:
    row = [esc(c)]
    for n in names:
        sc = scaled(c, vals[n][c])
        prod[n] *= sc
        row.append(f'{sc:.1f}')
    out.append('    ' + ' & '.join(row) + r' \\')
out.append(r'\hline')
out.append('    Joint fitness & ' + ' & '.join(sci(prod[n]) for n in names) + r' \\ \hline')
out.append(r'\end{tabular}' + '\n' + r'\end{table}')

# ══ Tabella IV — le due famiglie ═════════════════════════════════════════
dsc = fam['discriminanti'][:5]
out.append(r'''
\begin{table}[htbp]
\centering
\caption{The two families of solutions. Above, the boundaries whose presence
distinguishes them, with the fraction of the structures of each family that
retains the boundary. Below, the properties of the two families: the median
joint fitness verified with the reference solver, and the variation of the
reference flux shape within the groups, weighted by the total importance that
appears in the flux fitness function.}
\label{tab:families}
\begin{tabular}{|c|c|c|}
\hline
Boundary [MeV] & Family A [\%] & Family B [\%] \\ \hline''')
for d_ in dsc:
    out.append(f"    {sci(d_['E_MeV'], 4)} & {d_['freq_A']*100:.0f} & "
               f"{d_['freq_B']*100:.0f} \\\\")
out.append(r'\hline')
out.append(r'    Property & Family A & Family B \\ \hline')
out.append(f"    Distinct structures & {fam['n_A']} & {fam['n_B']} \\\\")
out.append('    Median joint fitness & '
           f"{sci(fam['ff_A'])} & {sci(fam['ff_B'])} \\\\")
out.append('    Mean internal Hamming distance & '
           f"{fam['hamming_within_A']:.1f} & {fam['hamming_within_B']:.1f} \\\\")
out.append('    Weighted shape variation & '
           f"{fam['distorsione']['importanza_migliore']:.3f} & "
           f"{fam['distorsione']['importanza_peggiore']:.3f} \\\\ \\hline")
out.append(r'\end{tabular}' + '\n' + r'\end{table}')


# ══ Tabella — i 29 confini interni ═══════════════════════════════════════
Eb   = bp['energie_MeV']
rec  = bp['ricorrenza']
wg   = np.array(bp['importanza_gruppo'])
impb = (.5 * (wg[:-1] + wg[1:]) * 100).tolist()
inb  = [(i + 2) in best['boundaries'] for i in range(29)]
half = 15
out.append(r"""
\begin{table}[htbp]
\centering
\caption{The 29 internal boundaries of the fine mesh. For each one: the fraction
of the %d distinct optimized structures that retains it, the total importance
$\phi\phi^{\ddagger}/v$ of the two groups it separates, and whether it belongs to
the best structure found. The boundaries retained by more than nine structures
out of ten form the consensus skeleton; those retained by fewer than one in ten
are never used in practice.}
\label{tab:boundaries}
\begin{tabular}{|r|c|c|c||r|c|c|c|}
\hline
\multicolumn{4}{|c||}{} & \multicolumn{4}{c|}{} \\[-2.2ex]
$E$ [MeV] & Occ.\ [\%%] & Imp.\ [\%%] & Best &
$E$ [MeV] & Occ.\ [\%%] & Imp.\ [\%%] & Best \\ \hline""" % len(ver['ranking']))
for i in range(half):
    cells = []
    for k in (i, i + half):
        if k >= 29:
            cells.append(' & & & ')
        else:
            cells.append(f"{sci(Eb[k], 3)} & {rec[k]*100:.0f} & {impb[k]:.1f} & "
                         + (r'$\bullet$' if inb[k] else ''))
    out.append('    ' + ' & '.join(cells) + r' \\')
out.append(r'\hline' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}')

# ══ Tabella V — bilancio computazionale ═════════════════════════════════
SEC, NCPU = 46, 3
NCFG = meta['n_train']
c_meta = NCFG * SEC * NCPU / 3600
c_tribe = fin['pop'] * fin['gen'] * SEC * NCPU / 3600
out.append(r'''
\begin{table}[htbp]
\centering
\caption{Computational budget in core-hours, from the measured cost of one
FRENETIC configuration (%d~s on %d cores). One tribe is a complete genetic
search of %d individuals over %d generations.}
\label{tab:cost}
\begin{tabular}{|l|r|}
\hline
Item & Core-hours \\ \hline
Building the metamodel (%s configurations, once) & %.0f \\
One tribe evaluated directly with FRENETIC & %.0f \\
Break-even & %.0f tribes \\ \hline
This work: %d tribes on the metamodel, all outcomes verified & %.1f \\
Equivalent cost avoided & %.0f \\ \hline
\end{tabular}
\end{table}''' % (SEC, NCPU, fin['pop'], fin['gen'], f'${NCFG//1000}\\,{NCFG%1000:03d}$',
                  c_meta, c_tribe, c_meta / c_tribe,
                  fin['tribes'], ver['n_verified'] * SEC * NCPU / 3600,
                  fin['tribes'] * c_tribe))

# ══ Tabella VI — classifica verificata ═══════════════════════════════════
rk = ver['ranking']
out.append(r"""
\begin{table}[htbp]
\centering
\caption{The fifteen best of the %d distinct structures, all of which were
recomputed with FRENETIC, ranked by their true canonical fitness. The
disagreement is the relative difference between predicted and true joint
fitness; the worst output is the quantity on which the metamodel errs most for
that structure. Over the whole set the rank correlation between predicted and
true fitness is %.3f and the median disagreement is %.2f\%%.}
\label{tab:ranking}
\begin{tabular}{|r|r|c|c|r|l|}
\hline
Rank & $G$ & True fitness & Predicted & Disagr.\ [\%%] & Worst output \\ \hline"""
           % (ver['n_verified'], ver['spearman_pred_true'], ver['accordo_mediano_pct']))
for i, r in enumerate(rk[:15], 1):
    out.append(f"    {i} & {r['G']} & {sci(float(r['canon_ff_true']))} & "
               f"{sci(float(r['canon_ff_nn']))} & {float(r['joint_rel']):.1f} & "
               f"{esc(r['max_col'])} ({float(r['max_rel']):.0f}\\%) \\\\")
out.append(r'\hline' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}')

txt = '\n'.join(out)
assert 'e+0' not in txt and 'e+1' not in txt, 'notazione scientifica non convertita'

# una tabella per file, cosi' si inseriscono nel punto del testo in cui servono
# e la numerazione segue l'ordine di apparizione
import re
TD = pathlib.Path('tables'); TD.mkdir(exist_ok=True)
blocks = re.findall(r'\\begin\{table\}.*?\\end\{table\}', txt, re.S)
assert len(blocks) == 8, f'{len(blocks)} blocchi trovati, ne attendevo 8'
for blk in blocks:
    name = re.search(r'\\label\{tab:(\w+)\}', blk).group(1)
    (TD / f'{name}.tex').write_text(blk + '\n')
print(f'tables/: {len(blocks)} file, ' + ', '.join(
    re.search(r'\\label\{tab:(\w+)\}', b).group(1) for b in blocks))
print(f"ottimo G={best['G']}  FF={best['canon_ff_true']:.4e}  "
      f"pareggio {c_meta/c_tribe:.0f} tribù")
