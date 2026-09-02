# Manoscritto — ottimizzazione della griglia energetica LFR

Sorgenti LaTeX dell'articolo per *Nuclear Science and Engineering*.
Questo repository è sincronizzato con Overleaf: si può lavorare da entrambe le
parti, ma vanno lette le note qui sotto.

## Compilazione

Su Overleaf non serve fare nulla: il documento principale è `paper.tex` e la
classe sta in `style/`. In locale:

```bash
./build.sh          # pdflatex -> bibtex -> pdflatex x2
```

## Cosa si modifica a mano e cosa no

| file | |
|---|---|
| `paper.tex` | **il testo** — si modifica a mano, da qui o da Overleaf |
| `references.bib` | bibliografia |
| `style/` | classe ANS e stile bibliografico, non toccare |
| `flowchart.tex` | schema TikZ del workflow |
| `figs/*.pdf` | **generate**, non modificare a mano |
| `tables/*.tex` | **generate**, non modificare a mano |

Figure e tabelle sono prodotte dagli script del progetto principale
(`../make_figures.py`, `../make_tables.py`) a partire dai risultati in
`../results/`. Sono versionate perché Overleaf non esegue Python: se cambiano i
risultati vanno rigenerate in locale e ricommittate.

```bash
python make_figures.py     # figs/fig_*.pdf
python make_tables.py      # tables/*.tex
python ../audit_numbers.py # le cifre in prosa coincidono con i risultati?
```

`audit_numbers.py` confronta i numeri scritti nel testo con quelli sul disco:
va lanciato dopo ogni cambio di modello o di dati, perché la prosa non si
aggiorna da sola.

## Sincronizzazione con Overleaf

Overleaf tira e spinge su questo repository. Il flusso è:

- **modifiche fatte su Overleaf** → *Menu → GitHub → Push Overleaf changes*
- **modifiche fatte qui** → `git push`, poi su Overleaf *Pull GitHub changes*

Overleaf non fa merge automatici: conviene non lavorare sullo stesso paragrafo
da entrambe le parti nello stesso momento, e allineare spesso.
