// Report GOLAND_water rev H (10/11/2026), scritto per Pietro: flutter vicino alla superficie libera con ala immagine
// nell'UVLM (casi E3 e B), repliche di NACA TR 1232 e van Walree 1999. Uso:
//   NODE_PATH=<node_modules con pptxgenjs> node deck_revH.js <cartella skill pptx>
// Legge dati_revG.json e le PNG di figure_revH.py, vgvf_revH.py, testP/naca1232 e testP/vanwalree.
// La bozza precedente, con riferimenti interni, è in deck_revH_bozza_interna.js.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(path.join(process.argv[2], "scripts", "apply_theme.js"));

const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "dati_revG.json"), "utf8"));
const OUT = path.join(HERE, "..", "GOLAND_water_sommergenza_LV_revH.pptx");
const DATA = "10 novembre 2026";
const FOOT = "GOLAND_water · Flutter vicino alla superficie libera · per Pietro · Revisione H · " + DATA;

const THEME = {
  name: "GOLAND_water revH",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "222222", lt1: "FFFFFF", dk2: "1B2A49", lt2: "EEF2F6",
    accent1: "E0622B", accent2: "1F9E89", accent3: "2E6FB7", accent4: "8E6BB8",
    accent5: "3C7A3C", accent6: "6B6B6B", hlink: "2E6FB7", folHlink: "8E6BB8",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "LV";
pres.title = "GOLAND_water – Flutter e sommergenza – Rev H";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITOLO",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.2, w: 11.9, h: 1.3, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.7, w: 11.9, h: 1.6, fontSize: 18, color: "C9D3E3", valign: "top" }, text: "" } },
    { placeholder: { options: { name: "note", type: "body", x: 0.7, y: 6.3, w: 11.9, h: 0.6, fontSize: 11, italic: true, color: "C9D3E3" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENUTO",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 10.5, h: 0.3, fontSize: 9, color: HEX.accent6 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: HEX.accent6, align: "right" },
});

const T = (s, txt, o) => s.addText(txt, Object.assign({ isTextBox: true, fontSize: 15, color: C.text1, valign: "top", margin: 0 }, o));
const H = (s, txt, x, y, w) => T(s, txt, { x, y, w, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: "Cambria" });
const bullets = (items, o) => items.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 }, o || {}) }));
const HDR = { bold: true, color: C.background1, fill: { color: HEX.dk2 }, fontSize: 12, align: "center", valign: "middle" };
const cell = (t, o) => ({ text: String(t), options: Object.assign({ fontSize: 12, color: C.text1, valign: "middle" }, o || {}) });
const table = (s, rows, o) => s.addTable(rows, Object.assign({ border: { type: "solid", pt: 0.75, color: "D5DCE6" }, fill: { color: "FFFFFF" }, margin: 0.05 }, o));
const card = (s, x, y, w, h, big, unit, label, color, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: name + " card" });
  s.addText([{ text: big, options: { fontSize: 40, bold: true, color } }, { text: " " + unit, options: { fontSize: 16, color: HEX.accent6 } }],
    { isTextBox: true, x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.85, fontFace: "Cambria", margin: 0, valign: "middle", objectName: name + " valore" });
  T(s, label, { x: x + 0.25, y: y + 1.0, w: w - 0.5, h: h - 1.1, fontSize: 13, color: C.text1, objectName: name + " etichetta" });
};
const regime = (s, y, tag, color, title, body, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 2.1, h: 1.05, fill: { color }, line: { color }, rectRadius: 0.08, objectName: name + " pastiglia" });
  T(s, tag, { x: 0.7, y, w: 1.9, h: 1.05, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", objectName: name + " regime" });
  T(s, [{ text: title, options: { bold: true, breakLine: true, fontSize: 14 } }, { text: body, options: { fontSize: 12.5 } }], { x: 2.9, y, w: 4.4, h: 1.05, valign: "middle", objectName: name + " testo" });
};
const f1 = (x, n) => Number(x).toFixed(n);
const pct = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1) + " %";

const R = D.rows; // h/c 2, 1, 0.5, 0.4
const r = (hc) => R.find((x) => Math.abs(x.hc - hc) < 1e-9);
const VU = D.uvlm;
const dG = 100 * (r(0.4).vp / r(2).vp - 1);
const dF = 100 * (r(0.4).v0 / r(2).v0 - 1);
const dO = 100 * (r(0.4).v3 / r(2).v3 - 1);


// ======================================================================================================
// Corpo del report rev H, scritto per Pietro: esperto del settore, ma non ha seguito il lavoro.
// Niente riferimenti interni (revisioni precedenti, nomi di prove, script): ogni cosa è spiegata dove compare.
// ======================================================================================================

// 1 ------------------------------------------------------------------ titolo
pres.addSection({ title: "Apertura" });
let s = pres.addSlide({ masterName: "TITOLO", sectionTitle: "Apertura" });
s.addText("Flutter di un'ala sommersa vicino alla superficie libera", { placeholder: "title" });
s.addText("La superficie libera nelle forze circolatorie, con un'ala immagine nell'UVLM: risultati, verifiche e confronto con la letteratura\nPer Pietro · LV\n" + DATA, { placeholder: "body" });
s.addText("Casi numerici dimostrativi (ala GOLAND_water in acciaio), non progetti. Revisione H.", { placeholder: "note" });

// 2 ------------------------------------------------------------------ in breve
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Apertura" });
s.addText("In breve", { placeholder: "title" });
T(s, bullets([
  "Domanda: quanto cambia la velocità di flutter V_f di un'ala sommersa quando si avvicina alla superficie libera?",
  "Finora la superficie libera entrava solo nella massa aggiunta e nello smorzamento di radiazione (BEM). Le forze circolatorie, che guidano il flutter, erano calcolate in fluido infinito",
  "Qui la superficie entra anche nelle forze circolatorie, con un'ala immagine nell'UVLM: è il limite di superficie a pressione nulla (φ = 0), valido ad alto numero di Froude",
  "Il modello d'immagine è verificato su due fonti esterne, entrambe stazionarie: NACA TR 1232 (misure in vasca) e la tesi di van Walree (metodo completo con onde, e misure)",
], { fontSize: 14 }), { x: 0.6, y: 1.3, w: 12.1, h: 2.9 });
card(s, 0.6, 4.4, 3.9, 2.3, pct(dG).replace(" %", ""), "%", "Caso realizzabile E3: V_f da h/c 2 a 0.4 (" + f1(r(2).vp, 1) + " → " + f1(r(0.4).vp, 1) + " kn). Con la superficie solo nel BEM era " + pct(dF), HEX.accent1, "E3");
card(s, 4.7, 4.4, 3.9, 2.3, "< 1", "%", "Effetto per sommergenze di una corda o più (h/c ≥ 1): trascurabile", HEX.accent2, "profondo");
card(s, 8.8, 4.4, 3.9, 2.3, "−43", "%", "Caso B, lento (Fn_h ≈ 2): V_f scende, ma qui contano le onde e il risultato va rifatto con la loro memoria", HEX.accent3, "B");

// 3 ------------------------------------------------------------------ il modello di calcolo
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Apertura" });
s.addText("Il modello di calcolo", { placeholder: "title" });
T(s, bullets([
  "Struttura: trave a sbalzo con flessione, torsione e accoppiamento flesso-torsionale; i primi 4 modi propri secchi formano la base modale",
  "Forze circolatorie: UVLM di SHARPy in fluido infinito → matrice modale Q(k), 4 × 4, in funzione della frequenza ridotta k",
  "Massa aggiunta A(ω; h) e smorzamento di radiazione B(ω; h): BEM (Capytaine) con superficie libera alla sommergenza h, a velocità di avanzamento nulla",
  "Somma senza doppio conteggio: dalla Q dell'UVLM si toglie la sua massa non circolatoria (in fluido infinito) e si mette quella del BEM (con la superficie)",
  "Velocità di flutter V_f: per ogni velocità V si trovano le radici p = σ + iω dell'equazione aeroelastica, con Q valutata alla frequenza della radice stessa; V_f è dove σ del ramo di flutter passa per zero. Smorzamento strutturale 1 %",
], { fontSize: 13.5 }), { x: 0.6, y: 1.3, w: 7.3, h: 5.5 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 1.3, w: 4.5, h: 5.5, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: "simboli card" });
T(s, "Simboli", { x: 8.45, y: 1.45, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.text2, fontFace: "Cambria" });
T(s, bullets([
  "h: sommergenza; c: corda; h/c sommergenza relativa",
  "Fn_h = U/√(g h): Froude di sommergenza; Fnc = U/√(g c): Froude di corda",
  "τ = U ω / g: parametro delle onde instazionarie (risonanza a τ = 1/4)",
  "k = ω c / (2U): frequenza ridotta",
  "σ/ω: smorzamento relativo della radice (σ > 0 instabile)",
  "φ = 0: superficie a pressione nulla, cioè potenziale nullo sulla superficie",
], { fontSize: 12 }), { x: 8.45, y: 1.95, w: 4.0, h: 4.7 });

// 4 ------------------------------------------------------------------ i due casi
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Apertura" });
s.addText("I due casi: E3 e caso B", { placeholder: "title" });
T(s, bullets([
  "Due ali rettangolari a sbalzo, in acciaio, completamente sommerse; stessa corda, stessi assi, stesso accoppiamento",
  "E3 è il caso realizzabile: prima frequenza propria f1 > 7 Hz (freccia statica in acqua di pochi mm) e sommergenze solo h/c ≥ 0.4, perché più in alto l'ala che vibra uscirebbe dall'acqua",
  "Il caso B è un esempio numerico, leggero e molto cedevole (f1 ≈ 1 Hz): non è costruibile, ma va in flutter a bassa velocità, dove le onde contano",
], { fontSize: 13 }), { x: 0.6, y: 1.3, w: 12.1, h: 1.9 });
table(s, [
  [cell("", HDR), cell("E3 – caso realizzabile", HDR), cell("Caso B – esempio numerico", HDR)],
  [cell("Corda × semi-apertura"), cell("294 × 1029 mm (AR 7)", { align: "center" }), cell("294 × 882 mm", { align: "center" })],
  [cell("Spessore relativo / materiale"), cell("12 % / acciaio", { align: "center" }), cell("20 % / acciaio", { align: "center" })],
  [cell("Asse elastico / baricentro"), cell("0.15c / 0.75c", { align: "center" }), cell("0.15c / 0.75c", { align: "center" })],
  [cell("EI / GJ (frazione della sezione piena), ψ"), cell("24 % / 3.0 %, ψ = 0.4", { align: "center" }), cell("0.087 % / 0.0217 %, ψ = 0.4", { align: "center" })],
  [cell("Modi secchi [Hz]"), cell("7.28 · 28.1 · 47.9 · 54.8", { align: "center", bold: true }), cell("1.08 · 4.36 · 7.27 · 8.73", { align: "center", bold: true })],
  [cell("Flutter, superficie solo nel BEM"), cell("torsione, ≈ 28 Hz in acqua, V_f ≈ 42.4–42.7 kn", { align: "center" }), cell("flesso-torsionale, ≈ 4.3 Hz, V_f ≈ 8.3 kn", { align: "center" })],
  [cell("Fn_h al flutter"), cell("9–22: alto Froude", { align: "center", color: HEX.accent5, bold: true }), cell("≈ 2: zona delle onde", { align: "center", color: HEX.accent1, bold: true })],
], { x: 0.6, y: 3.3, w: 12.1, colW: [3.5, 4.3, 4.3], rowH: 0.4, fontSize: 12 });
T(s, "ψ: accoppiamento flesso-torsionale, K₃₄ = ψ·√(EI·GJ). In entrambi l'asse elastico sta davanti al centro aerodinamico: niente divergenza.", { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 11, color: HEX.accent6 });

// 5 ------------------------------------------------------------------ superficie libera e immagine
pres.addSection({ title: "Metodo" });
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Metodo" });
s.addText("La superficie libera nelle forze circolatorie", { placeholder: "title" });
regime(s, 1.35, "Fn_h ≫ 1 e τ ≫ 1", HEX.accent1, "Pressione nulla, φ = 0",
  "Immagine a +2h con la stessa circolazione (effetto biplano): la portanza cala. È il modello usato qui", "phi0");
regime(s, 2.6, "Fn_h ≈ 1–2 o τ ≈ 1/4", HEX.accent3, "Onde di gravità",
  "Serve la funzione di Green con la memoria delle onde (Kelvin/Havelock, van Walree)", "onde");
regime(s, 3.85, "Fn_h ≪ 1", HEX.accent6, "Parete rigida",
  "Immagine di circolazione opposta (effetto suolo): la portanza sale", "parete");
T(s, "Fra i due limiti la portanza dipende da Froude e sommergenza insieme (Nishiyama 1965; Hough e Moran 1969).", { x: 0.6, y: 5.1, w: 6.7, h: 0.6, fontSize: 12, color: HEX.accent6 });
H(s, "E3 al punto di flutter", 7.7, 1.35, 5.0);
const Rrows = [[cell("h/c", HDR), cell("Fn_h", HDR), cell("ω²h/g", HDR), cell("τ", HDR)]];
R.forEach((x) => Rrows.push([cell(x.hc, { align: "center" }), cell(f1(x.fnh, 1), { align: "center" }), cell(f1(x.w2h, 0), { align: "center" }), cell(f1(x.tau, 0), { align: "center" })]));
table(s, Rrows, { x: 7.7, y: 1.85, w: 5.0, colW: [1.1, 1.3, 1.3, 1.3], rowH: 0.42 });
T(s, bullets([
  "E3: Fn_h 9–22, τ ≈ 400. Le onde sono cortissime rispetto alla sommergenza: il limite φ = 0 vale con ampio margine",
  "Caso B: Fn_h ≈ 2, τ ≈ 9–12. Siamo al bordo della zona delle onde",
], { fontSize: 13 }), { x: 7.7, y: 4.15, w: 5.0, h: 2.3 });

// 6 ------------------------------------------------------------------ implementazione
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Metodo" });
s.addText("L'ala immagine nell'UVLM di SHARPy", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "schema_immagine.png"), x: 0.4, y: 1.3, w: 7.0, h: 3.08, altText: "Ala vera a profondità h, superficie libera a pressione nulla, ala immagine traslata a +2h con la stessa circolazione" });
T(s, bullets([
  "Si aggiunge una seconda superficie portante, copia dell'ala vera traslata di 2h verso l'alto, con la propria scia",
  "La copia si muove esattamente come l'ala vera: stessa traslazione verticale, rotazione e incidenza (niente traslazione spuria 2h·θ)",
  "Le forze sulla copia non tornano alla struttura: servono solo a imporre φ = 0 sulla superficie",
  "La circolazione della copia non è imposta, la risolve l'UVLM; per simmetria risulta uguale a quella dell'ala vera, come richiede φ = 0 (verificato a 1e-13 nella replica NACA)",
  "L'equazione di flutter non cambia: Q resta 4 × 4, cambia solo Q(k) → Q(k; h)",
], { fontSize: 13 }), { x: 7.7, y: 1.3, w: 5.0, h: 5.6 });
T(s, bullets([
  "Costo: l'immagine raddoppia i pannelli. Sulla griglia fine (32 pannelli in corda) la riduzione del modello UVLM supera la memoria disponibile (> 10 GB)",
  "Soluzione: Q(h) = Q_fine + [Q_rada,immagine(h) − Q_rada,immagine(immagine lontana)], con griglia rada di 16 pannelli in corda. Incertezza circa ±0.2 kn su V_f (confronto con 8 pannelli)",
], { fontSize: 12 }), { x: 0.6, y: 4.6, w: 6.8, h: 2.3 });

// 7 ------------------------------------------------------------------ verifiche numeriche
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Metodo" });
s.addText("Verifiche numeriche", { placeholder: "title" });
table(s, [
  [cell("controllo", HDR), cell("esito", HDR)],
  [cell("Immagine molto lontana (2h = 29 m)"), cell("Portanza statica dell'ala vera invariata (9.592 N); correzione nulla a k = 0")],
  [cell("Portanza statica con / senza immagine a h/c 2 · 1 · 0.5 · 0.4"), cell("SHARPy 0.950 · 0.893 · 0.802 · 0.767; VLM stazionario a ferri di cavallo, scritto a parte e indipendente da SHARPy, con l'immagine come secondo reticolo a +2h: " + R.map((x) => f1(x.f3d, 3)).join(" · ") + ": scarto ≤ 0.3 %")],
  [cell("Massa non circolatoria da h/c 2 a 0.4"), cell("UVLM con immagine −7…−9 %, BEM −8…−11 %: stesso verso, quindi la superficie non è contata due volte")],
  [cell("Griglia della correzione immagine (16 contro 8 pannelli in corda)"), cell("Scarto 7–8 % della correzione a h/c 0.4, cioè circa ±0.2 kn su V_f")],
  [cell("Riferimento della correzione"), cell("Immagine lontana invece del fluido infinito: cancella l'errore della riduzione del modello su un sistema di dimensione doppia (0.3 % a k = 8.5, 1 % a k = 24)")],
  [cell("Caso B, griglia (16 contro 8 pannelli) a h/c 0.4"), cell("V_f 4.76 contro 4.71 kn")],
], { x: 0.6, y: 1.4, w: 12.1, colW: [4.4, 7.7], rowH: [0.45, 0.6, 0.9, 0.6, 0.6, 0.75, 0.5] });

// 8 ------------------------------------------------------------------ fonti
pres.addSection({ title: "Letteratura" });
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("Chi modella la superficie con un'immagine", { placeholder: "title" });
const SI = (t, o) => cell(t, Object.assign({ fontSize: 11 }, o || {}));
table(s, [
  [cell("fonte", HDR), cell("modello della superficie", HDR), cell("regime", HDR), cell("replicato qui?", HDR)],
  [SI("Wadlin, Shuford, McGehee 1955 (NACA TR 1232)", { bold: true }), SI("Velocità supercritica: ferro di cavallo immagine sopra la superficie, con la stessa rotazione, w letta ai 3/4 di corda"), SI("stazionario, ala rigida; prove in vasca A = 4 e 10"), SI("Sì: equazioni del rapporto e confronto con le misure", { bold: true, color: HEX.accent5 })],
  [SI("van Walree 1999 (tesi TU Delft)", { bold: true }), SI("UVLM con funzione di Green transitoria: immagine biplano + memoria delle onde"), SI("stazionario e instazionario, qualunque Froude"), SI("Sì per il limite φ = 0, contro le sue curve e le misure di Wilson 1983; la memoria delle onde no", { bold: true, color: HEX.accent5 })],
  [SI("Thiart 1994; 2001 (R&D Journal SAIMechE)", { bold: true }), SI("Immagine + potenziale ondoso, superficie linearizzata a Froude qualunque"), SI("stazionario"), SI("No. Il suo risultato: in stazionario la scia lontana vede un'immagine di segno opposto a ogni Froude. Per la scia oscillante non vale, perché varia sulla scala U/ω ≪ U²/g")],
  [SI("Kaplan 1951; Falch 1991 (citati da van Walree)", { bold: true }), SI("Vortice con immagine biplano (Kaplan anche con termine di gravità)"), SI("moto in onde"), SI("No: non letti in originale")],
], { x: 0.6, y: 1.35, w: 12.1, colW: [2.6, 3.9, 2.2, 3.4], rowH: [0.4, 0.95, 0.95, 1.0, 0.7] });
T(s, bullets([
  "Nessuna fonte trovata confronta l'immagine con misure di flutter: per il moto instabile la validità di φ = 0 poggia su τ ≫ 1 e Fn_h ≫ 1",
  "Nota sul §6 di GINO_teoria_metodo_v4: in 2D il fattore di portanza vale 1 − (c/h)²/16, non /32, perché la downwash dell'immagine va letta ai 3/4 di corda (come in NACA TR 1232). Le percentuali stimate lì vanno circa raddoppiate",
], { fontSize: 12.5 }), { x: 0.6, y: 5.85, w: 12.1, h: 1.15 });

// 9 ------------------------------------------------------------------ NACA: modello
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("NACA TR 1232: lo stesso modello, riprodotto", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "naca1232_fig15.png"), x: 0.4, y: 1.25, w: 12.5, h: 4.1, altText: "Rapporto delle pendenze di portanza a2/a1 contro f/c per A = 10, 6, 4: equazioni NACA, VLM con immagine imposta, VLM con ala clone risolta, curva 2D ed esperimenti" });
T(s, bullets([
  "Il rapporto usa un solo ferro di cavallo e una sola immagine imposta: le sue equazioni (6)–(7) sono riprodotte esattamente",
  "Con molti pannelli (lo stesso VLM stazionario indipendente da SHARPy), immagine imposta e ala copia risolta (il nostro metodo) coincidono a 1e-13",
], { fontSize: 13 }), { x: 0.6, y: 5.45, w: 6.0, h: 1.5 });
T(s, bullets([
  "Il reticolo fine perde meno portanza del ferro di cavallo singolo a piccola sommergenza; le misure stanno fra i due",
  "f = sommergenza nella notazione NACA (qui h); a2/a1 = pendenza della portanza vicino alla superficie / in fluido infinito",
], { fontSize: 13 }), { x: 6.9, y: 5.45, w: 5.8, h: 1.5 });

// 10 ----------------------------------------------------------------- NACA: SHARPy contro misure
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("SHARPy con ala immagine contro le misure NACA", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "sharpy_naca1232_fig15.png"), x: 0.4, y: 1.25, w: 12.5, h: 4.1, altText: "a2/a1 contro f/c: SHARPy con ala immagine, equazioni NACA, curva 2D ed esperimenti in vasca" });
table(s, [
  [cell("scarto dai 16 punti sperimentali [punti di a2/a1]", HDR), cell("medio", HDR), cell("rms", HDR), cell("max", HDR)],
  [cell("SHARPy con immagine (stesso codice del flutter, ala rigida, α = 0.1°)"), cell("+1.1", { align: "center" }), cell("2.7", { align: "center", bold: true }), cell("5.3", { align: "center" })],
  [cell("Equazioni NACA (6)–(7)"), cell("−0.4", { align: "center" }), cell("2.1", { align: "center" }), cell("3.7", { align: "center" })],
  [cell("dispersione fra fonti sperimentali (A = 10)"), cell("", {}), cell("2.4", { align: "center" }), cell("", {})],
], { x: 0.6, y: 5.4, w: 7.4, colW: [4.7, 0.9, 0.9, 0.9], rowH: 0.36, fontSize: 11 });
T(s, bullets([
  "f/c ≥ 1: dentro la dispersione delle misure",
  "f/c 0.5–0.85: le misure perdono 3–5 punti di portanza più di φ = 0 (spessore 18 %, viscosità, Froude finito)",
  "Serve α piccolo: a 1° le due scie inclinate non sono più speculari (errore 0.5 punti)",
], { fontSize: 11.5 }), { x: 8.3, y: 5.4, w: 4.4, h: 1.6 });

// 11 ----------------------------------------------------------------- van Walree: funzione di Green
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("Van Walree 1999: immagine più memoria delle onde", { placeholder: "title" });
T(s, bullets([
  "Tesi TU Delft (promotore G. Kuiper), UVLM per aliscafi in moto stazionario e instazionario, con superficie libera linearizzata",
  "Funzione di Green transitoria di un vortice a intensità impulsiva (eq. 4.35, nella forma di Newman 1985)",
  "Nel metodo lineare l'immagine entra nella matrice d'influenza (eq. 4.56–4.61): Biot–Savart del vortice e della sua immagine",
  "p. 33: per Fn_h > 10 la velocità non conta più e basta l'immagine biplano; per 1 < Fn_h < 2 dominano le onde e la portanza cala di più; per Fn_h < 1 sale sopra il fluido infinito",
  "Validato su misure con la superficie: Wilson 1983 (stazionario e in onde), Kyozuka 1992 (profilo 2D in heave). Il flutter non è trattato",
], { fontSize: 13 }), { x: 0.6, y: 1.3, w: 6.0, h: 5.5 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.3, w: 5.7, h: 5.5, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: "green card" });
T(s, "Funzione di Green (eq. 4.35)", { x: 7.3, y: 1.45, w: 5.1, h: 0.4, fontSize: 16, bold: true, color: C.text2, fontFace: "Cambria" });
T(s, "G = G° + Gf = 1/R − 1/R₀ + Gf", { x: 7.3, y: 2.0, w: 5.1, h: 0.5, fontSize: 18, bold: true, color: HEX.accent1, fontFace: "Cambria", objectName: "formula green" });
T(s, "Gf = 2 ∫₀^∞ [1 − cos(√(gk)(t − τ))] e^{k(z+ζ)} J₀(kr) dk", { x: 7.3, y: 2.6, w: 5.1, h: 0.4, fontSize: 13, color: C.text1 });
T(s, bullets([
  "G°: vortice più immagine biplano (R₀ = distanza dal punto speculare). Si annulla sulla superficie: è esattamente la nostra ala immagine",
  "Gf: memoria delle onde generate dalla storia del moto. È nulla per Fn → ∞; per Fn → 0 vale −2 volte l'immagine, che si inverte e dà la parete rigida",
  "Il nostro modello corrisponde a G° da solo: il limite ad alto Froude del metodo di van Walree",
], { fontSize: 12.5 }), { x: 7.3, y: 3.15, w: 5.1, h: 3.5 });

// 12 ----------------------------------------------------------------- replica van Walree
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("Van Walree: il limite φ = 0 riprodotto", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "replica_vanwalree_fig3_9.png"), x: 0.4, y: 1.2, w: 7.6, h: 4.16, altText: "C_Lalpha(h/c)/C_Lalpha(h/c=4) contro Fn_h per AR 6: curve di van Walree, misure di Wilson, asintoto della nostra immagine phi=0; bande di Fn_h al flutter di E3 e del caso B" });
const VWR = [[cell("h/c", HDR), cell("φ = 0, nostro", HDR), cell("van Walree a Fn_h 10", HDR), cell("van Walree, Fn_h → ∞", HDR)]];
[[2, 0.971, 0.973, 0.977], [1, 0.911, 0.908, 0.913], [0.5, 0.816, 0.797, 0.808], [0.25, 0.706, 0.651, 0.674]].forEach((x) =>
  VWR.push([cell(x[0], { align: "center" }), cell(f1(x[1], 3), { align: "center", bold: true }), cell(f1(x[2], 3), { align: "center" }), cell(f1(x[3], 3), { align: "center" })]));
table(s, VWR, { x: 8.3, y: 1.3, w: 4.4, colW: [0.7, 1.1, 1.3, 1.3], rowH: 0.38, fontSize: 11 });
T(s, "Tesi, Fig. 3.9a: ala rettangolare AR 6, 4 × 16 pannelli, rapporto della pendenza di portanza a quella a h/c = 4. Curve e misure lette dalla figura (±0.01); valore a Fn_h → ∞ estrapolato come r∞ − a/Fn_h².", { x: 8.3, y: 3.3, w: 4.4, h: 0.8, fontSize: 10, color: HEX.accent6 });
T(s, bullets([
  "Il nostro φ = 0 coincide con il valore a cui tendono le sue curve: entro 1 punto per h/c ≥ 0.5, 3 punti a h/c 0.25",
  "Le misure di Wilson seguono le sue curve: a Fn_h 6–8 stanno 4–7 punti sotto φ = 0, per l'effetto residuo della gravità",
], { fontSize: 12 }), { x: 8.3, y: 4.15, w: 4.4, h: 2.0 });
T(s, bullets([
  "E3 (Fn_h 9–22): l'effetto della gravità vale al massimo 1.4 punti a h/c 0.5. Il limite φ = 0 è confermato da un metodo indipendente e dalle misure",
  "Caso B (Fn_h ≈ 2): siamo sul minimo delle curve. A h/c 0.5 la portanza stazionaria vera è 0.61, contro 0.82 di φ = 0 (−25 %)",
], { fontSize: 12.5 }), { x: 0.6, y: 5.45, w: 7.4, h: 1.5 });

// 13 ----------------------------------------------------------------- peso della memoria delle onde
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Letteratura" });
s.addText("Quanto pesano le onde sul moto oscillante", { placeholder: "title" });
H(s, "Van Walree, AR 6, h/c 1: con le onde / solo immagine", 0.6, 1.3, 7.2);
table(s, [
  [cell("Fnc", HDR), cell("portanza stazionaria (Tab. 5.2)", HDR), cell("ampiezza, heave ampio (Tab. 5.3)", HDR), cell("ampiezza, in onde (Tab. 5.4)", HDR), cell("fase, in onde", HDR)],
  [cell("1.6", { align: "center", bold: true }), cell("0.885", { align: "center" }), cell("1.600", { align: "center" }), cell("0.989", { align: "center" }), cell("−7.4°", { align: "center", bold: true, color: HEX.accent1 })],
  [cell("3.2", { align: "center", bold: true }), cell("0.957", { align: "center" }), cell("1.121", { align: "center" }), cell("0.987", { align: "center" }), cell("−1.8°", { align: "center" })],
  [cell("6.4", { align: "center", bold: true }), cell("0.991", { align: "center" }), cell("1.037", { align: "center" }), cell("0.994", { align: "center" }), cell("−0.5°", { align: "center" })],
], { x: 0.6, y: 1.8, w: 7.2, colW: [0.8, 1.6, 1.9, 1.6, 1.3], rowH: [0.6, 0.42, 0.42, 0.42], fontSize: 11 });
T(s, bullets([
  "Tab. 5.3: heave di 0.75 corde a k = 0.25, con l'ala che sale fino a 0.25 c dalla superficie. È un effetto non lineare, lontano dalle piccole ampiezze del flutter",
  "Tab. 5.4, piccole ampiezze: l'ampiezza della portanza cambia meno dell'1.3 %, ma la fase fino a 7° a Fnc 1.6",
  "Conclusione dell'autore: a Froude di crociera le onde contano poco; per Fnc fra 1 e 4 vanno incluse",
], { fontSize: 12.5 }), { x: 0.6, y: 3.9, w: 7.2, h: 2.6 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 1.3, w: 4.5, h: 5.5, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: "casi card" });
T(s, "Dove stanno i nostri casi", { x: 8.45, y: 1.45, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.text2, fontFace: "Cambria" });
table(s, [
  [cell("", HDR), cell("Fnc", HDR), cell("k", HDR), cell("τ", HDR)],
  [cell("E3", { bold: true }), cell("≈ 13", { align: "center" }), cell("1.1", { align: "center" }), cell("≈ 400", { align: "center" })],
  [cell("caso B, h/c 1", { bold: true }), cell("2.2", { align: "center" }), cell("1.2", { align: "center" }), cell("11", { align: "center" })],
  [cell("caso B, h/c 0.4", { bold: true }), cell("1.4", { align: "center" }), cell("2.1", { align: "center" }), cell("9", { align: "center" })],
  [cell("van Walree, Tab. 5.3", { italic: true }), cell("1.6–6.4", { align: "center" }), cell("0.25", { align: "center" }), cell("1.3–20", { align: "center" })],
], { x: 8.45, y: 1.95, w: 4.0, colW: [1.6, 0.8, 0.7, 0.9], rowH: 0.38, fontSize: 11 });
T(s, bullets([
  "E3: ben oltre il campo in cui le onde contano",
  "Caso B: Fnc nel campo in cui vanno incluse. τ è alto, lontano dalla risonanza τ = 1/4, quindi l'ampiezza dovrebbe cambiare poco; ma la fase può spostarsi di alcuni gradi, e nel caso B il ramo instabile attraversa lo zero in modo piatto",
], { fontSize: 12 }), { x: 8.45, y: 4.0, w: 4.0, h: 2.75 });

// 14 ----------------------------------------------------------------- risultati E3
pres.addSection({ title: "Risultati" });
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Risultati" });
s.addText("E3: velocità di flutter con l'immagine", { placeholder: "title" });
card(s, 0.6, 1.35, 3.9, 1.75, f1(r(0.4).vp, 2), "kn", "V_f a h/c 0.4 (" + pct(100 * (r(0.4).vp / VU - 1)) + " rispetto all'ala senza superficie libera, " + f1(VU, 2) + " kn)", HEX.accent1, "vf04");
card(s, 4.7, 1.35, 3.9, 1.75, pct(dG).replace(" %", ""), "%", "variazione di V_f fra h/c 2 e 0.4", HEX.accent1, "dG");
card(s, 8.8, 1.35, 3.9, 1.75, f1(r(2).fp, 1) + "–" + f1(r(0.4).fp, 1), "Hz", "frequenza di flutter (torsione), quasi costante con la sommergenza", HEX.dk2, "ff");
const G1 = [[cell("", HDR)].concat(R.map((x) => cell("h/c " + x.hc, HDR)))];
G1.push([cell("V_f [kn]", { bold: true })].concat(R.map((x) => cell(f1(x.vp, 2), { align: "center", bold: true }))));
G1.push([cell("f_f [Hz]", { bold: true })].concat(R.map((x) => cell(f1(x.fp, 2), { align: "center" }))));
G1.push([cell("rispetto all'ala senza superficie", { bold: true })].concat(R.map((x) => cell(pct(100 * (x.vp / VU - 1)), { align: "center" }))));
G1.push([cell("Fn_h", { bold: true })].concat(R.map((x) => cell(f1(x.fnh, 1), { align: "center" }))));
table(s, G1, { x: 0.6, y: 3.35, w: 12.1, colW: [3.3, 2.2, 2.2, 2.2, 2.2], rowH: 0.42 });
T(s, bullets([
  "Sotto mezza corda la superficie riduce le forze circolatorie e alza V_f di 2–3 kn; a h/c ≥ 1 l'effetto è sotto l'1 %",
  "A h/c 2 V_f scende di 0.3 kn rispetto alla sola massa aggiunta: la curva di smorzamento è piatta vicino a σ = 0, e piccole differenze spostano l'attraversamento",
], { fontSize: 14 }), { x: 0.6, y: 5.65, w: 12.1, h: 1.3 });

// 15 ----------------------------------------------------------------- confronto con / senza
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Risultati" });
s.addText("E3: dove entra la superficie libera", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "confronto_vf.png"), x: 0.5, y: 1.25, w: 6.6, h: 4.4, altText: "V_f contro h/c: nessuna superficie libera, superficie solo nella massa aggiunta, superficie anche nelle forze circolatorie" });
const CF = [[cell("h/c", HDR), cell("nessuna superficie", HDR), cell("solo massa aggiunta (BEM)", HDR), cell("anche forze circolatorie", HDR), cell("differenza", HDR)]];
R.forEach((x) => CF.push([cell(x.hc, { align: "center" }), cell(f1(VU, 2), { align: "center", color: HEX.accent6 }), cell(f1(x.v0, 2), { align: "center" }),
  cell(f1(x.vp, 2), { align: "center", bold: true }), cell((x.vp - x.v0 >= 0 ? "+" : "−") + f1(Math.abs(x.vp - x.v0), 2) + " kn", { align: "center", bold: x.hc < 1, color: x.hc < 1 ? HEX.accent1 : HEX.dk1 })]));
CF.push([cell("2 → 0.4", { align: "center", bold: true }), cell("0 %", { align: "center", color: HEX.accent6 }), cell(pct(dF), { align: "center" }), cell(pct(dG), { align: "center", bold: true }), cell("", {})]);
table(s, CF, { x: 7.3, y: 1.35, w: 5.4, colW: [0.8, 1.15, 1.15, 1.2, 1.1], rowH: [0.6, 0.42, 0.42, 0.42, 0.42, 0.42], fontSize: 11 });
T(s, "V_f in kn, ramo di torsione fra 20 e 40 Hz.", { x: 7.3, y: 4.3, w: 5.4, h: 0.4, fontSize: 11, color: HEX.accent6 });
T(s, bullets([
  "Con la superficie solo nella massa aggiunta la sommergenza vale " + pct(dF) + "; con l'immagine " + pct(dG) + ", dieci volte tanto",
  "Ad alto Froude l'effetto della superficie sta nella portanza circolatoria, non nella massa aggiunta",
  "Ignorarlo è conservativo: vicino alla superficie il flutter vero è più alto",
], { fontSize: 13 }), { x: 7.3, y: 4.75, w: 5.4, h: 2.2 });

// 16 ----------------------------------------------------------------- smorzamento
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Risultati" });
s.addText("E3: smorzamento del ramo di flutter", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "confronto_sigma.png"), x: 0.9, y: 1.3, w: 11.5, h: 5.29, altText: "sigma/omega del ramo di flutter contro V a quattro sommergenze: continua superficie solo nella massa aggiunta, tratteggiata anche nelle forze circolatorie" });
T(s, "Continue: superficie solo nella massa aggiunta, curve quasi sovrapposte. Tratteggiate: anche nelle forze circolatorie. A h/c 2 e 1 le curve restano vicine; a h/c 0.5 e 0.4 l'immagine smorza il ramo e sposta l'attraversamento a 44.7 e 45.8 kn. Pallini: σ = 0.", { x: 0.9, y: 6.5, w: 11.5, h: 0.55, fontSize: 11, color: HEX.accent6 });

// 17 ----------------------------------------------------------------- perché sale
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Risultati" });
s.addText("E3: perché V_f sale", { placeholder: "title" });
card(s, 0.6, 1.35, 3.9, 1.9, "−23", "%", "portanza stazionaria a h/c 0.4 (k = 0): l'immagine toglie downwash all'ala vera", HEX.accent3, "stat");
card(s, 4.7, 1.35, 3.9, 1.9, "−8…−12", "%", "|Q| circolatoria alla frequenza di flutter, a h/c 0.4", HEX.accent1, "flutter");
card(s, 8.8, 1.35, 3.9, 1.9, "−7…−9", "%", "massa non circolatoria UVLM a h/c 0.4, come quella del BEM (−8…−11 %)", HEX.accent2, "massa");
T(s, bullets([
  "Il flutter di E3 è torsionale ed è alimentato dalle forze circolatorie: se queste calano, serve più pressione dinamica per destabilizzare il ramo",
  "L'effetto biplano dipende dalla frequenza: alla frequenza di flutter il calo è circa metà di quello stazionario. Un fattore di riduzione stazionario applicato a tutta Q sovrastimerebbe V_f di 0.5–1 kn",
  "La massa aggiunta cala allo stesso modo in UVLM e BEM: la sostituzione la gestisce senza doppio conteggio, e il cambio di V_f viene dalla parte circolatoria",
  "La frequenza di flutter resta a 28.2–28.4 Hz; con la sola massa aggiunta saliva da 28.1 a 28.9 Hz",
], { fontSize: 14.5 }), { x: 0.6, y: 3.6, w: 12.1, h: 3.3 });

// 18 ----------------------------------------------------------------- caso B
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Risultati" });
s.addText("Caso B: con l'immagine V_f scende", { placeholder: "title" });
s.addImage({ path: path.join(HERE, "vgvf_casoB.png"), x: 0.4, y: 1.2, w: 7.6, h: 3.65, altText: "Caso B: sigma/omega e frequenza di tutte le radici contro V a h/c 2, 1, 0.5, 0.4, senza immagine (grigio) e con immagine (arancio)" });
const CB = [[cell("h/c", HDR), cell("solo massa aggiunta", HDR), cell("con immagine", HDR), cell("f_f [Hz]", HDR), cell("Fn_h", HDR)]];
[[2, 8.26, 8.16, 4.26, 1.7], [1, 8.28, 7.27, 4.79, 2.2], [0.5, 8.34, 5.23, 5.38, 2.2], [0.4, 8.36, 4.76, 5.50, 2.3]].forEach((x) =>
  CB.push([cell(x[0], { align: "center" }), cell(f1(x[1], 2), { align: "center" }), cell(f1(x[2], 2), { align: "center", bold: true, color: x[0] < 1 ? HEX.accent1 : HEX.dk1 }), cell(f1(x[3], 2), { align: "center" }), cell(f1(x[4], 1), { align: "center" })]));
table(s, CB, { x: 8.3, y: 1.3, w: 4.4, colW: [0.6, 1.15, 1.05, 0.85, 0.75], rowH: 0.4, fontSize: 11 });
T(s, "V_f in kn, primo ramo instabile fra 2.5 e 6 Hz.", { x: 8.3, y: 3.4, w: 4.4, h: 0.4, fontSize: 10.5, color: HEX.accent6 });
T(s, bullets([
  "Al contrario di E3: −12 % a h/c 1, −43 % a h/c 0.4",
  "Diventa instabile un altro ramo, a ~5.5 Hz: senza immagine resta smorzato (σ/ω −0.03…−0.045) fra 2 e 7.6 kn",
  "L'attraversamento è piatto (σ/ω ≈ −0.01 già a 3–4 kn): V_f è molto sensibile",
], { fontSize: 12.5 }), { x: 8.3, y: 3.85, w: 4.4, h: 3.1 });
T(s, bullets([
  "Ma Fn_h ≈ 2 è proprio dove, secondo van Walree, le onde riducono di più la portanza (−25 % rispetto a φ = 0 a h/c 0.5)",
  "Il verso del risultato è plausibile, il valore no: va rifatto con la memoria delle onde",
], { fontSize: 12.5 }), { x: 0.6, y: 5.0, w: 7.4, h: 1.95 });

// 19 ----------------------------------------------------------------- limiti
pres.addSection({ title: "Conclusioni" });
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Conclusioni" });
s.addText("Limiti", { placeholder: "title" });
table(s, [
  [cell("limite", HDR), cell("conseguenza", HDR)],
  [cell("Superficie a pressione nulla (φ = 0): solo l'immagine, senza memoria delle onde"), cell("Valido per Fn_h ≫ 1 e τ ≫ 1 (E3: Fn_h ≥ 9, τ ≈ 400), confermato da van Walree e dalle misure di Wilson. Per Fn_h ≈ 1–2 servono le onde")],
  [cell("Caso B fuori dal campo di validità"), cell("V_f scende fino a −43 %, ma con Fn_h ≈ 2 il valore non è affidabile")],
  [cell("Verifiche esterne solo stazionarie"), cell("NACA TR 1232 (rms 2.7 punti) e van Walree/Wilson (entro 1 punto); nessun dato di flutter vicino alla superficie trovato")],
  [cell("Correzione immagine su griglia rada"), cell("Circa ±0.2 kn su V_f; la griglia fine con l'immagine non sta in memoria")],
  [cell("BEM a velocità di avanzamento nulla"), cell("A τ ≈ 400, con o senza avanzamento, si tende alla pressione nulla")],
  [cell("Superficie piana e indeformata (lineare)"), cell("Niente rottura d'onda né ventilazione, da verificare a parte a h/c 0.4")],
], { x: 0.6, y: 1.4, w: 12.1, colW: [4.6, 7.5], rowH: [0.45, 0.85, 0.6, 0.75, 0.6, 0.6, 0.6] });

// 20 ----------------------------------------------------------------- conclusioni
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Conclusioni" });
s.addText("Conclusioni", { placeholder: "title" });
T(s, bullets([
  "Ad alto Froude la superficie libera agisce sulle forze circolatorie (effetto biplano), non sulla massa aggiunta",
  "Nel caso realizzabile E3, V_f va da " + f1(r(2).vp, 1) + " kn (h/c 2) a " + f1(r(0.4).vp, 1) + " kn (h/c 0.4): " + pct(dG) + ", contro " + pct(dF) + " con la superficie solo nella massa aggiunta",
  "A h/c ≥ 1 l'effetto è sotto l'1 %; conta sotto mezza corda",
  "Calcolare le forze circolatorie in fluido infinito è conservativo: sottostima V_f vicino alla superficie",
  "Il modello d'immagine è il limite ad alto Froude del metodo di van Walree e il modello supercritico di NACA TR 1232: riprodotti entrambi, entro 1–3 punti",
  "Nel caso B (Fn_h ≈ 2) V_f scende fino a −43 %, ma lì servono le onde",
], { fontSize: 14 }), { x: 0.6, y: 1.4, w: 7.4, h: 5.4 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.4, y: 1.4, w: 4.3, h: 4.6, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: "passo card" });
T(s, "Possibili passi", { x: 8.7, y: 1.6, w: 3.8, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: "Cambria" });
T(s, bullets([
  "Mettere l'immagine nella matrice d'influenza dell'UVLM, come van Walree: niente pannelli doppi, griglia fine possibile",
  "Per il caso B, aggiungere la memoria delle onde (eq. 4.35 di van Walree)",
  "Fonti ancora da leggere: van Walree 2000 (MARIN) e i rapporti DTIC sul flutter di aliscafi vicino alla superficie",
], { fontSize: 13.5 }), { x: 8.7, y: 2.1, w: 3.8, h: 3.8 });

// 21 ----------------------------------------------------------------- bibliografia
s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: "Conclusioni" });
s.addText("Bibliografia", { placeholder: "title" });
const REF = [
  ["Wadlin, K.L., Shuford, C.L., McGehee, J.R. (1955). ", "A theoretical and experimental investigation of the lift and drag characteristics of hydrofoils at subcritical and supercritical speeds.", " NACA Report 1232."],
  ["van Walree, F. (1999). ", "Computational methods for hydrofoil craft in steady and unsteady flow.", " Tesi di dottorato, TU Delft."],
  ["van Walree, F. (2000). ", "Validation of an unsteady vortex lattice method for hydrofoil craft.", " MARIN (non ancora letto)."],
  ["Wilson, M.B. (1983). ", "Experimental determination of low Froude number hydrofoil performance in calm water and in regular waves.", " 20th American Towing Tank Conference, Hoboken."],
  ["Thiart, G.D. (1994). ", "Numerical lifting line theory for a hydrofoil near a free surface.", " R&D Journal (SAIMechE) 10(1), 18–23."],
  ["Thiart, G.D. (2001). ", "Generalized vortex lattice method for prediction of hydrofoil characteristics.", " R&D Journal (SAIMechE) 17(2), 35–46."],
  ["Newman, J.N. (1985). ", "The evaluation of free-surface Green functions.", " 4th Int. Conf. on Numerical Ship Hydrodynamics, Washington, 4–19."],
  ["Nishiyama, T. (1965). ", "Lifting-surface theory of a fully submerged hydrofoil.", " J. Ship Research 9(1)."],
  ["Hough, G.R., Moran, J.P. (1969). ", "Froude number effects on two-dimensional hydrofoils.", " J. Ship Research 13, 53–60."],
  ["Faltinsen, O.M. (2005). ", "Hydrodynamics of High-Speed Marine Vehicles,", " cap. 6. Cambridge University Press."],
];
const refRuns = [];
REF.forEach((x, i) => {
  refRuns.push({ text: x[0], options: { bold: true } });
  refRuns.push({ text: x[1], options: { italic: true } });
  refRuns.push({ text: x[2], options: { breakLine: i < REF.length - 1, paraSpaceAfter: 6 } });
});
T(s, refRuns, { x: 0.6, y: 1.3, w: 12.1, h: 5.65, fontSize: 12.5 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("scritto", OUT);
})();
