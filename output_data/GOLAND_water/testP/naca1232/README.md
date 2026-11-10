# Riproduzione di NACA TR 1232 (figure 14–15), 06/11/2026

Script: `riproduzione_naca1232.py`, in puro Python; usa Biot-Savart da `testO/fattore_immagine.py`.

Esce:
- `naca1232_fig15.png`;
- `naca1232.json`.

Rapporto originale: `../letteratura/Wadlin_Shuford_McGehee_1955_NACA_TR1232.pdf`. Il modello è a p. 4 del rapporto (fig. 12) e le equazioni sono a pp. 10–11.

## Cosa fa il rapporto

- **Ala vera:** un solo ferro di cavallo al quarto di corda; tangenza in un solo punto, ai 3/4 di corda sulla mezzeria; ala rigida e moto stazionario.
- **Immagine:** un ferro di cavallo a +2f con la stessa rotazione. Non è un'ala: non ha geometria, rigidezza, massa, moto né condizione di tangenza propria. Ha la circolazione dell'ala per costruzione: Γ1/Γ2 = 1 + W6/W5, eq. (8).
- **Validazione:** prove in vasca (A = 4 e 10) e dati da letteratura (A = 6 e 10), a velocità supercritica.

## Risultati

1. **Le equazioni NACA (6)–(7) sono riprodotte esattamente** dal VLM ridotto a un solo ferro di cavallo (stesse 4 cifre).
2. **Immagine imposta e ala clone risolta coincidono a 1e-13.** Nel clone risolto l'ala ha incognite proprie e la stessa tangenza, come in SHARPy con `image_surface.py`. Γ_clone = Γ_ala non è un'ipotesi: segue dalla simmetria. La w indotta da un vortice orizzontale è pari in z, quindi ala e clone si vedono a vicenda allo stesso modo.
3. **Il VLM fine riduce meno la portanza del ferro di cavallo singolo** a piccola sommergenza: A = 4, f/c 0.5 dà 0.812 contro 0.754.
4. **Gli esperimenti cadono fra i due modelli.** A f/c ≥ 1 sono più vicini al VLM; a f/c 0.6–0.85 sono 2–4 punti sotto il VLM. Possibili cause:
   - spessore 18 % (NACA 65₃-418);
   - viscosità;
   - Froude finito;
   - lettura a occhio della figura (±0.01).
5. **Il 2D sta sopra il 3D:** in 3D l'immagine della scia toglie altra portanza (effetto biplano), come trovato nel test O.

## Cosa non copre

Il rapporto è stazionario. Per le forze instabili la stessa simmetria vale se il clone ha la stessa cinematica e la propria scia: è ciò che fa SHARPy, quindi Γ_clone(t) = Γ_ala(t). La validità fisica di φ = 0 per il moto oscillante richiede però τ = Uω/g ≫ 1, condizione che il rapporto non discute.

## SHARPy con l'ala immagine contro NACA 1232 (06/11/2026)

Script: `sharpy_naca1232.py` (78 casi, pochi secondi l'uno) e `confronto_sharpy_naca1232.py`. Escono `sharpy_naca1232.json` e `sharpy_naca1232_fig15.png`.

**Impostazione:**
- ala rettangolare piana rigida, corda 1, A = 4, 6, 10;
- template Goland con `add_image_surfaces` del test P, senza trave: `BeamLoader → AerogridLoader → StaticUvlm`;
- scia a ferro di cavallo infinita;
- incidenza data dalla corrente, α = 0.1°;
- griglie M8 (8 × 8A pannelli) e M16;
- a2/a1 = L(f/c) / L(fluido infinito).

**Controlli:**
- **Codice:** SHARPy M8 contro il VLM Python con la stessa pannellatura, scarto ≤ 0.0005. Γ_imm/Γ_ala ≥ 0.9998.
- **Griglia:** M8 contro M16, scarto ≤ 0.0004.
- **Effetto di α:** a α = 1° le due scie, entrambe inclinate di α e non speculari, rompono la simmetria e lo scarto sale a 0.005. Quei risultati sono in `_sharpy_naca1232_alpha1.json`. Per il moto armonico a piccola ampiezza vale il caso α → 0.

**Contro i 16 punti sperimentali** (scarto in punti di a2/a1):

| | medio | rms | max |
|---|---|---|---|
| SHARPy con immagine | +1.1 | 2.7 | 5.3 |
| NACA eq. (6)–(7) | −0.4 | 2.1 | 3.7 |
| dispersione fra le fonti sperimentali (A = 10, f/c 0.84 e 3.84) | | | 2.4 |

**Lettura:**
- A f/c ≥ 1, SHARPy sta dentro la dispersione sperimentale (−2.1…+1.2 punti). Unica eccezione A = 6, f/c 1 (rif. 9, +5.3), un punto isolato.
- A f/c 0.53–0.85, SHARPy sovrastima a2/a1 di 3–5 punti in 5 casi su 6: le misure perdono più portanza dell'immagine φ = 0. Le cause possibili sono spessore 18 %, viscosità e Froude finito, già elencate sopra.
- Il modello NACA a un solo ferro di cavallo è più vicino alle misure, ma per compensazione: è più grossolano e sta sotto il VLM convergente.
- **Per il flutter:** se la tendenza stazionaria si trasferisce, a h/c ≤ 1 la riduzione vera della Q circolatoria è un po' più forte di quella di SHARPy. Il +9.1 % di V_f di E3 andrebbe allora letto come stima per difetto. Resta da verificare: E3 ha un'altra pianta ed è un caso instabile.
