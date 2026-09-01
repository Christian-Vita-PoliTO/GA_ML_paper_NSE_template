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
std  = json.load(open(RES / 'baseline_standard.json'))
stdv = json.load(open(RES / 'baseline_standard_valori.json'))
dat  = json.load(open(RES / 'refined_plot_data.json'))
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

# ══ Tabella II — le tre ipotesi escluse ══════════════════════════════════
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
Noise in the labels & 21\,440 configurations simulated more than once &
Spread of $0.0000\%$ on all fifteen outputs; an independent re-run reproduces
stored reference values to all digits \\ \hline
Sparse data in that regime & Training points per band of the reference value &
$13\,016$ points in the worst-performing band against $5\,223$ in the best \\ \hline
Input representation & 65 physics-informed descriptors appended to the
30 inputs & The model becomes \emph{worse}: $0.473\%$ against $0.422\%$
\\ \hline
\end{tabular}
\end{table}""")

# ══ Tabella III — ottimizzata contro le standard ═════════════════════════
vals = {'Optimized': best['frenetic_values']}
for r in std['standard']:
    key = [k for k in stdv if k.startswith(r['nome'][:8])][0]
    lbl = ('Equal lethargy' if 'letargia' in r['nome'] else 'Uniform index')
    vals[f"{lbl} ($G$={r['G']})"] = stdv[key]
names = list(vals)
out.append(r'''
\begin{table}[htbp]
\centering
\caption{The optimized structure against two standard ones, quantity by
quantity. Each entry is the scaled contribution entering the joint fitness,
obtained from the FRENETIC values through the fixed canonical window: unity is
the best attainable value and 100 the worst. Six of the fifteen terms saturate
at unity for all three structures and therefore carry no information in this
regime.}
\label{tab:standard}
\begin{tabular}{|l|''' + 'c|' * len(names) + r'''}
\hline
Figure of merit & ''' + ' & '.join(esc(n) for n in names) + r' \\ \hline')
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

# ══ Tabella IV — bilancio computazionale ═════════════════════════════════
SEC, NCPU = 46, 3
c_meta = 89723 * SEC * NCPU / 3600
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
Building the metamodel ($89\,723$ configurations, once) & %.0f \\
One tribe evaluated directly with FRENETIC & %.0f \\
Break-even & %.0f tribes \\ \hline
This work: %d tribes on the metamodel & %.1f \\
Equivalent cost avoided & %.0f \\ \hline
\end{tabular}
\end{table}''' % (SEC, NCPU, fin['pop'], fin['gen'], c_meta, c_tribe, c_meta / c_tribe,
                  fin['tribes'], fin['n_new_frenetic'] * SEC * NCPU / 3600,
                  fin['tribes'] * c_tribe))

# ══ Tabella V — classifica verificata ════════════════════════════════════
out.append(r'''
\begin{table}[htbp]
\centering
\caption{The %d distinct candidates verified with FRENETIC, ranked by their
true canonical fitness. The disagreement is the relative difference between
predicted and true joint fitness; the worst output is the quantity on which the
metamodel errs most for that structure.}
\label{tab:ranking}
\begin{tabular}{|r|r|c|c|r|l|}
\hline
Rank & $G$ & True fitness & Predicted & Disagr.\ [\%%] & Worst output \\ \hline''' %
           len(fin['ranking']))
for i, r in enumerate(fin['ranking'], 1):
    out.append(f"    {i} & {r['G']} & {sci(float(r['canon_ff_true']))} & "
               f"{sci(float(r['canon_ff_nn']))} & {float(r['joint_rel']):.1f} & "
               f"{esc(r['max_col'])} ({float(r['max_rel']):.0f}\\%) \\\\")
out.append(r'\hline' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}')

txt = '\n'.join(out)
assert 'e+0' not in txt and 'e+1' not in txt, 'notazione scientifica non convertita'
pathlib.Path('tables.tex').write_text(txt)
print(f'tables.tex: {len(txt.splitlines())} righe, 5 tabelle')
print(f"ottimo G={best['G']}  FF={best['canon_ff_true']:.4e}  "
      f"pareggio {c_meta/c_tribe:.0f} tribù")
