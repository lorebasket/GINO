# Van Walree 1999: lettura della tesi e replica del limite φ = 0 (10/11/2026)

Fonte: F. van Walree, *Computational methods for hydrofoil craft in steady and unsteady flow*, tesi di dottorato, TU Delft, 1999 (`../letteratura/1999-PhD-Walree.pdf`, 376 pagine). Il rapporto MARIN 2000 non è ancora disponibile. La tesi non tratta il flutter.

## Cosa dice sulla superficie libera

**Funzione di Green** (eq. 4.35, p. 215; forma di Newman 1985 per una sorgente impulsiva, adattata a un vortice):

G = G° + Gf = 1/R − 1/R₀ + 2 ∫₀^∞ [1 − cos(√(g k)(t − τ))] e^{k(z+ζ)} J₀(k r) dk

- **G° = "vortex plus biplane image".** È esattamente la nostra ala immagine φ = 0: G° si annulla sulla superficie.
- **Gf** è la memoria ondosa. Per Fn → ∞ vale zero (p. 243). Per Fn → 0 tende a −2 volte i termini d'immagine, cioè l'immagine si inverte e si ottiene la parete rigida.
- Nel metodo lineare (eq. 4.61) G° entra come matrice d'influenza: Biot–Savart di vortice e immagine sopra la superficie (eq. 4.56–4.58). È l'"immagine nel nucleo" che noi avevamo come passo facoltativo.

**Regimi, p. 33** (sulla teoria 2D di Hough e Moran 1969):
- Fn_h > 10: la dipendenza dalla velocità sparisce e basta l'immagine biplano.
- 1 < Fn_h < 2: le onde dominano e la riduzione di portanza è massima.
- Fn_h < 1 e sommergenza piccola: portanza sopra quella in fluido infinito.

**Peso di Gf.** Ala rettangolare AR 6, h/c 1, rapporti con / senza Gf.

| Fnc | Tab. 5.2: C_z stazionario | Tab. 5.3: ampiezza C_z, heave k 0.25 ampio (0.75 c), non lineare | Tab. 5.4: ampiezza C_z, onde incidenti | Tab. 5.4: Δfase |
|---|---|---|---|---|
| 1.6 | 0.885 | 1.600 | 0.989 | −7.4° |
| 3.2 | 0.957 | 1.121 | 0.987 | −1.8° |
| 6.4 | 0.991 | 1.037 | 0.994 | −0.5° |

- Conclusione dell'autore (conclusioni del §5.2): per Fnc da crociera Gf è piccolo sia stazionario sia oscillante. Per moti di heave ampi (z_a/h > 0.15) Gf conta.
- Con Fnc fra 1 e 4 (foil poco sommersi e lenti) Gf va incluso.

**Validazione sperimentale con la superficie:**
- Wilson 1983 (AR 6, stazionario, Fig. 3.9);
- Kyozuka 1992 (2D in heave, h/c 0.9, Fnc 0.71, Fig. 5.32c–d), con un buco di portanza a τ = 1/4;
- Wilson 1983 in onde (AR 6, h/c 0.5, Fig. 5.33).

## Replica: `replica_vanwalree.py`

Esce `replica_vanwalree.json` e `replica_vanwalree_fig3_9.png`.

- **Impostazione:** Fig. 3.9a (p. 143), AR 6, N 4 × M 16, tip inset 0.25, rapporto C_Lα(h/c)/C_Lα(h/c = 4) contro Fn_h.
- **Digitalizzazione:** a mano dalla scansione, ±0.01.
- **Il nostro modello:** VLM con immagine concorde (stessa costruzione di testO, naca1232 e dell'ala clone SHARPy). Non dipende da Fn_h, quindi va confrontato con l'asintoto.
- **Estrapolazione delle curve di van Walree:** r(Fn) = r∞ − a/Fn², sui punti a Fn_h 6 e 10.

| h/c | φ = 0 nostro (N4M16 / N8M48) | van Walree a Fn_h 10 | van Walree estrapolato | scarto |
|---|---|---|---|---|
| 2 | 0.971 / 0.971 | 0.973 | 0.977 | −0.006 |
| 1 | 0.911 / 0.912 | 0.908 | 0.913 | −0.002 |
| 0.5 | 0.816 / 0.817 | 0.797 | 0.808 | +0.008 |
| 0.25 | 0.706 / 0.709 | 0.651 | 0.674 | +0.032 |

**Lettura:**
- **L'asintoto di van Walree è il nostro φ = 0:** entro 1 punto per h/c ≥ 0.5, entro 3 punti a h/c 0.25. Lì l'estrapolazione 1/Fn² è la parte più incerta.
- **Le misure di Wilson seguono le curve di van Walree.** A Fn_h 6–8 stanno 3.6 punti (h/c 0.5) e 7 punti (h/c 0.25) sotto φ = 0: è l'effetto di gravità residuo, che va come 1/Fn_h².
- **E3 (Fn_h 9–22):** la correzione di gravità vale a/Fn_h² ≤ 1.4 punti a h/c 0.5 (a ≈ 1.1) e scende sotto 0.3 punti a Fn_h 22. Il limite φ = 0 è confermato da un metodo indipendente e da misure.
- **Caso B (Fn_h ≈ 2):** siamo sul minimo delle curve. A h/c 0.5 la portanza stazionaria vera è 0.61, contro 0.82 di φ = 0 (−25 %).
  - Per la parte oscillante del flutter (τ 9–12, k 1.2–2.1, Fnc 1.4–2.2) la Tab. 5.4 suggerisce correzioni di ampiezza piccole (≈ 1 %), ma fasi fino a 7°. Questi casi però hanno k più bassi.
  - Il caso B resta quindi da rifare con Gf.

## Non risolto

La Fig. 5.5b (C_z contro Fnc, AR 6, h/c 1, α 1°, N4M16) dà C_z ≈ 0.0735 a Fnc 6.4. È il nostro valore in fluido infinito (0.0733), non quello con immagine (0.0658). A parità di caso questo non torna con la sua Fig. 3.9a (rapporto 0.91 a h/c 1). Non ho trovato la ragione nel testo: diversa normalizzazione o diversa impostazione. Uso solo la Fig. 3.9a, che riporta anche le misure.
