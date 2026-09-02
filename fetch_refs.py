"""Scarica le voci BibTeX da CrossRef: campi esatti, non dedotti."""
import json, subprocess, urllib.parse, re, sys

# chiave -> DOI noto  oppure  titolo da cercare
ITEMS = {
 'prevGA':      '10.1080/00295639.2024.2446130',
 'freneticBench':'10.1140/epjp/s13360-020-00171-8',
 'frenetic':    'A full-core coupled neutronic/thermal-hydraulic code for the modeling of lead-cooled nuclear fast reactors',
 'massoneGA':   '10.1016/j.anucene.2017.03.022',
 'simmerGA':    'SIMMER extension for multigroup energy structure search using genetic algorithm with different fitness functions',
 'collapsing':  '10.1016/j.anucene.2025.111550',
 'sfrSA':       'Optimization of multi-group energy structures for diffusion analyses of sodium-cooled fast reactors assisted by simulated annealing',
 'mlClassify':  'Classification of group structures for a multigroup collision probability model using machine learning',
 'pbfa':        'Heuristic optimization of group structure using Physics-Based Fitness Approximation',
 'psoThermal':  'An investigation for an optimized neutron energy-group structure in thermal lattices using Particle Swarm Optimization',
 'scwr':        'Determination of the optimal few-energy group structure for the Canadian Super Critical Water-cooled Reactor',
 'arcGA':       'A genetic algorithm to optimize the multi-group structure for the neutronic analyses of the ARC fusion reactor',
 'fnsSurrogate':'10.3389/fenrg.2022.874194',
 'coreLoading': '10.1080/00295639.2025.2598170',
 'saeaSurvey':  '10.1007/s40747-024-01465-5',
 'pymoo':       '10.1109/ACCESS.2020.2990567',
 'alfred':      '10.1016/j.pnucene.2013.11.011',
 'contributon': 'Optimization of energy-group structure for LWR high-fidelity neutronics calculation based on the contributon theory',
}

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout

def doi_of(title):
    q = urllib.parse.quote(title)
    r = sh(f'curl -s --max-time 40 "https://api.crossref.org/works?query.bibliographic={q}&rows=1"')
    try:
        it = json.loads(r)['message']['items'][0]
        return it['DOI'], it['title'][0]
    except Exception:
        return None, None

out, report = [], []
for key, val in ITEMS.items():
    if val.startswith('10.'):
        doi, found = val, '(DOI noto)'
    else:
        doi, found = doi_of(val)
        if doi is None:
            report.append((key, 'NON TROVATO', val)); continue
    bib = sh(f'curl -sL --max-time 40 -H "Accept: application/x-bibtex" "https://doi.org/{doi}"')
    if not bib.lstrip().startswith('@'):
        report.append((key, 'BIBTEX FALLITO', doi)); continue
    bib = re.sub(r'@(\w+)\{[^,]+,', lambda m: f'@{m.group(1)}{{{key},', bib.strip(), count=1)
    out.append(bib)
    t = re.search(r'title\s*=\s*[{"]+(.+?)[}"]+,', bib, re.S)
    report.append((key, doi, (t.group(1)[:70] if t else found)))

open('crossref.bib', 'w').write('\n\n'.join(out) + '\n')
print(f'{len(out)}/{len(ITEMS)} voci scaricate\n')
for k, d, t in report:
    print(f'  {k:<15} {d:<42} {re.sub(chr(10)+r"\s*"," ",str(t))[:60]}')
