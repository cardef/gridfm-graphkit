# Piano A — Simmetrie della mappa AC power flow come bias induttivi per Grid Foundation Models

*Bozza di piano di ricerca — settembre 2026 · rivista il 30 settembre e il 2 ottobre 2026*

> **Revisione del 2 ottobre 2026** (replica del PoC sul cluster, 46 run, e F0 sui 36 checkpoint GENCO pubblicati;
> verdetti in `experiments/ac_pf_symmetries/results/abacus/VERDICTS.md` e `results/genco/VERDICTS.md`):
> (1) H2 nella forma del 30/09 è contraddetta: sui checkpoint pubblicati EE_S3 predice l'errore zero-shot anche dopo la
> canonicalizzazione. (2) H4: il vantaggio zero-shot di canon del PoC (un seed) non si separa su 5 seed, e il
> rappresentante S3 non conta a questa scala. (3) M0+Aug sull'intero range non addestra con GNS, neanche con la loss nel
> frame del campione o senza loss fisica. (4) Il rifit del normalizzatore sul target è un'azione S3 esatta:
> |ΔRMSE| ≤ EE_S3(k) è un'identità. (5) I dataset pubblici sono tutti N-1: M0+Canon ≡ M0 in distribuzione vale solo a
> topologia fissa. (6) L'errore sugli angoli di nodo è per lo più un offset comune rispetto al REF (R7).

> **Revisione del 30/09/2026** (dopo la revisione critica e il PoC in `experiments/ac_pf_symmetries/`):
> (1) S4 corretta: in AC l'inversione from/to *scambia* i flussi, non li nega; il decoder antisimmetrico di M4 imponeva
> perdite nulle; la regola dei trasformatori include il riporto dell'impedenza τ²z.
> (2) La scelta casuale del bus di riferimento non è una simmetria finché REF è anche lo slack.
> (3) Ogni simmetria del catalogo ammette una canonicalizzazione esatta in forma chiusa (Prop. 5): la domanda
> "equivarianza esatta vs augmentation" diventa "quale rappresentazione della sezione canonica", con M0+Canon come
> baseline di ogni confronto.
> (4) Senza sfasatori S2 si riduce a S1 sui dati (Prop. 6): H3 passa da ipotesi a proposizione.
> (5) H1, H2, H4 riformulate; aggiunto R7.
> (6) EE per canale e relativo all'RMSE; M0+Aug con loss nel frame del campione; zero-shot senza etichette del target.
> (7) M1 con proiezione di Hodge; condizione di iniettività per M3.

---

## 1. In una frase

Catalogare le trasformazioni che lasciano invariante la soluzione del power flow come un unico gruppo di gauge,
mostrare che ogni convenzione di modellazione ne è un gauge fixing e che ogni simmetria ammette una canonicalizzazione
esatta (quindi la sua rottura è eliminabile a costo zero), usare l'errore di equivarianza come test di consistenza delle
pipeline, e spostare la domanda sperimentale su ciò che la simmetria non decide: quale rappresentazione della sezione
canonica trasferisce a reti non viste.

## 2. Motivazione e posizionamento nella roadmap GridFM

La roadmap identifica il transfer zero-shot a reti non viste come il collo di bottiglia che separa uno "shared grid
model" (livello 1) da un "transferable GridFM" (livello 2), e riporta che le reti a valori complessi migliorano la
predizione OOD degli angoli di circa due ordini di grandezza *attribuendo il guadagno al bias induttivo e non alla
capacità* (§4.2). Quale bias sia non è stabilito: con θ_ref fisso nei dati la simmetria di fase U(1) non può esserne la
causa (Prop. 5); restano l'algebra bilineare complessa di S = V ⊙ conj(YV) e le coordinate rettangolari (equazioni
quadratiche, niente wrap degli angoli). Il piano separa le due spiegazioni (M2 contro il suo controllo split, §6).

Il piano copre:
- **§4.2 (architetture)**: canonicalizzazione esatta come baseline gratuito; layer equivarianti e scelte di
  rappresentazione confrontati a parità di simmetria.
- **§4.4 (pretraining e transfer)**: le convenzioni (base p.u., riferimento, orientamento) si canonicalizzano per campione,
  quindi il pretraining multi-rete non ha bisogno di augmentation per esse; ciò che resta dello shift tra reti vive in
  quantità invarianti (R/X, carico, topologia).
- **§4.6 (introspezione)**: l'*errore di equivarianza* di un modello addestrato è un test di consistenza della pipeline
  calcolabile senza etichette (scova input mai variati e normalizzatori congelati); non è un predittore dell'errore OOD (H2).

## 3. Stato dell'arte e gap

| Lavoro | Cosa fa | Cosa manca |
|---|---|---|
| Okoyomon, Yaniv, Goebel 2025 (arXiv:2509.25158) | Ablazione controllata di tre bias (PF-loss, reti complesse, riformulazione residuale) su ENGAGE; reti complesse battono DC-PF di un ordine di grandezza sugli angoli OOD | Il bias "reti complesse" non è separato in simmetria di fase vs algebra complessa; nessun catalogo; nessuna misura dell'equivarianza |
| Stiasny & Cremer 2026, *Residual Power Flow* (arXiv:2601.09533) | Angoli di ramo invece di angoli di bus: elimina il bus di riferimento; mostra che la relazione angolo-potenza diventa quasi lineare e più facile da apprendere | 9 bus, MLP; la KVL diventa un residuo; gli autori stessi indicano architetture graph-based come passo successivo |
| Dogoulis et al. 2025, KCLNet (arXiv:2506.12902); Flow-Attentional GNN (arXiv:2506.06127) | Vincoli di conservazione (KCL) imposti strutturalmente | Non trattano simmetrie di fase, scala, orientamento |
| GNN gauge-equivarianti U(1) (arXiv:2511.16062); Cohen et al. 2019; Favoni et al. 2021 (L-CNN) | Teoria e layer per equivarianza di gauge locale su grafi e reticoli | Mai applicati a reti elettriche |
| Kaba et al. 2023; Puny et al. 2022 (frame averaging); Dym et al. 2024 | Canonicalizzazione e frame: una funzione equivariante equivale a una funzione arbitraria su una sezione; per alcuni gruppi (es. rotazioni di nuvole di punti) una canonicalizzazione continua non esiste | Mai applicati a reti elettriche; qui tutti i gruppi ammettono frame continui (Prop. 5) |
| Puech et al. 2026, GENCO (arXiv:2608.09921) | Backbone eterogeneo PF/OPF/SE, decoder fisici, correzione iterativa; dataset multi-rete | Angoli di nodo rispetto allo slack con θ_ref ≡ 0 nei dati (S1 mai esercitata), dati tutti N-1; il normalizzatore fissa per rete una base data-driven (P95 delle iniezioni): è già un'azione S3 esatta per rete, ma la statistica legge Qg e il Pg dello slack, che nel PF sono output; il tap entra come feature nel verso from→to: S4 esatta sulle linee, rotta sui trasformatori, debolmente (36 checkpoint pubblicati: EE/RMSE 0.05–0.37, EE relativo 3·10⁻⁵–3·10⁻⁴); la base MVA pesa molto di più (EE_S3 relativo 0.2–0.8 a k = 0.1) |

**Gap:** non esistono (i) un catalogo formale delle simmetrie della mappa AC-PF come gruppo di gauge, con i frame che le
canonicalizzano, (ii) una diagnostica dell'errore di equivarianza per modelli addestrati, (iii) una separazione
controllata tra ciò che è attribuibile alla simmetria (eliminabile per canonicalizzazione) e ciò che è attribuibile alla
rappresentazione (angoli di nodo vs di ramo, algebra reale vs complessa, normalizzazione globale vs locale).

## 4. Domande di ricerca e ipotesi

**RQ1.** Quali trasformazioni lasciano invariante/equivariante la soluzione AC-PF, quali sono rotte dalle scelte di
rappresentazione standard, e con quale frame si ripristinano esattamente?

**RQ2.** L'errore di equivarianza dei modelli addestrati predice il loro errore su reti non viste, e in quali condizioni?

**RQ3.** A parità di simmetria esatta (M0+Canon), quali scelte di rappresentazione (angoli di ramo con proiezione di
Hodge, algebra complessa, feature adimensionali locali, rappresentante lungo l'orbita S3) migliorano efficienza dati e
generalizzazione OOD?

- **H1 (rappresentazione).** Almeno una scelta di rappresentazione (M1–M3) migliora lo zero-shot su reti non viste
  rispetto a M0+Canon di più della variabilità tra seed. Il vincolo di simmetria in sé non contribuisce sulla sezione
  (Prop. 5): ogni guadagno di M1–M4 su M0+Canon è per costruzione un guadagno di rappresentazione. Fuori sezione
  (convenzioni diverse) M0+Canon eguaglia per costruzione i modelli equivarianti e batte l'augmentation, che copre solo
  il range campionato (PoC: EE_S1 6·10⁻³ dentro il range, ≈ 1 fuori). Con GNS l'augmentation sull'intero range non
  addestra affatto (replica: VA 5–6° contro 0.2–0.3°), né con la loss nel frame del campione né senza loss fisica: il
  confronto con l'augmentation va fatto su range mild.
- **H2 (diagnostica).** Dentro una classe di modelli non equivarianti, EE_g calcolato senza etichette sul target ordina i
  modelli per errore OOD. Sui 36 checkpoint GENCO pubblicati (zero-shot tra reti IEEE) EE_S3,VM(k = 10) ha ρ parziale
  0.76 [0.57, 0.87] con l'errore VM su 81 coppie tenute fuori, ma anche 0.69 [0.50, 0.82] con l'errore che resta dopo la
  canonicalizzazione: la forma del 30/09 ("ρ ≈ 0 senza componente lungo le orbite") è contraddetta. Lettura più semplice:
  EE misura una sensibilità generica agli input spostati, non la parte d'errore che la simmetria rimuove; è un segnale
  d'allarme senza etichette, non una misura del guadagno di simmetria. Da testare dentro una classe al variare
  dell'intensità dell'augmentation. I modelli canonicalizzati hanno EE ≡ 0 per costruzione e vanno esclusi: su una
  popolazione mista ρ misurerebbe la classe, non l'equivarianza.
- **H3 (S2, residua).** Senza sfasatori S2 si riduce a S1 sui dati (Prop. 6): nessun guadagno possibile. Con sfasatori
  reali (PEGASE, ACTIVSg), M2 non batte la stessa rete complessa con gauge di Coulomb, perché gli sfasatori stanno in
  anelli e le direzioni di gauge non compaiono nei dati.
- **H4 (sezione S3; nulla per la parte di simmetria).** Una canonicalizzazione adattiva per campione, con frame dei soli
  input, recupera tutti i benefici attribuibili a S1–S4 (lo prevede la Prop. 5); resta la scelta del rappresentante
  lungo l'orbita S3. La famiglia s_a = (P95 delle iniezioni di input)^a · (mean|Y|)^(1−a), a ∈ [0, 1], è tutta esatta;
  ipotesi: la scelta di a cambia l'errore zero-shot più della variabilità tra seed. PoC, seed 0, normalizzatore della
  sorgente: canon (allineato in ammettenza) aveva VM 0.012 contro 0.023 di M0 su case30 e 0.017 contro 0.032 su case57.
  Esteso su dati indipendenti con 5 seed: 0.016 ± 0.003 contro 0.016 ± 0.007 su case30, 0.025 ± 0.010 contro
  0.033 ± 0.013 su case57, intervalli sovrapposti; a ∈ {0, 0.5, 1} non sposta l'errore zero-shot oltre la variabilità
  tra seed, mentre a > 0 raddoppia l'errore in distribuzione. Esatta resta l'indipendenza di canon dalla base scelta
  dal normalizzatore (Prop. 5 su un modello addestrato; verificata anche sui checkpoint pubblicati, 7.5·10⁻⁵), mentre
  M0 si sposta fino a 2×. A questa scala (una rete sorgente, due target piccoli) H4 non è sostenuta; resta aperta per
  il multi-rete (R2/R3). Fissare 100 MVA è solo una sezione mal scelta per il cross-dominio, non un test di H4.

## 5. Formalizzazione

Rete G = (N, B), matrice di ammettenza Y nella convenzione MATPOWER (trasformatore ideale al lato from, tap τ_ij,
sfasamento φ_ij): Y_ff = (y_s + j b/2)/τ², Y_ft = −y_s e^{jφ}/τ, Y_tf = −y_s e^{−jφ}/τ, Y_tt = y_s + j b/2; setpoint u
(P, Q ai bus PQ; P, V ai PV; V, θ allo slack). Mappa soluzione Φ: (G, Y, u) ↦ x = (V, θ), con S = diag(V)·conj(YV).

Simmetrie (ciascuna una proposizione con dimostrazione di due righe nel paper; tutte verificate numericamente):

- **S1 — U(1) globale.** g_α: θ_i ↦ θ_i + α ∀i. Flussi e iniezioni sono invarianti; Φ è equivariante rispetto a
  (θ_ref in ingresso, θ in uscita). Il riferimento angolare è un gauge solo se è separato dal ruolo di slack
  (bilanciamento): spostare lo slack cambia chi assorbe le perdite, quindi la soluzione. "Bus di riferimento casuale" è
  una simmetria solo come ri-riferimento degli output (θ_i − θ_r) a slack invariato; la codifica GridFM, in cui REF è
  anche lo slack, non la esprime.
- **S2 — gauge U(1) locale.** Per ψ ∈ ℝᴺ: V_i ↦ e^{jψ_i} V_i e φ_ft ↦ φ_ft + ψ_f − ψ_t. Iniezioni e flussi sono
  invarianti. Gli sfasatori sono una *connessione* sugli archi, e il message passing complesso con trasportatori
  U_ij = e^{jφ_ij} è gauge-covariante; S1 è il caso ψ costante. Invarianti: le olonomie Σ_ciclo φ, una per anello
  indipendente (|B| − |N| + 1). La differenza d'arco invariante è θ_f − θ_t − φ_ft.
- **S3 — scala (analisi dimensionale).** Cambio di base MVA k > 0: (P, Q, Y, shunt) ↦ (P, Q, Y, shunt)/k con V_pu
  invariato; Φ è invariante (PG, QG equivarianti). Forma locale: S = diag(V)·conj(YV) è invariante sotto V ↦ CV,
  Y ↦ C̄⁻¹YC⁻¹, con C diagonale in (ℂ*)ᴺ. La fase di C è S2; il modulo è il cambio di base di tensione per livello, con
  i tap come connessione ℝ₊ (τ_ft ↦ τ_ft s_f/s_t, impedenze riportate e setpoint/limiti di tensione riscalati).
  **Invarianti fisici, non toccati da S3:** R/X, carico P|Z|/V², rapporto di charging, topologia. Lo shift
  trasmissione → distribuzione vive in questi.
- **S4 — orientamento dei rami (Z₂ per arco).** Invertire from/to **scambia** (S_ft, S_tf) e nega la differenza
  angolare; i flussi AC non si negano, perché S_ft + S_tf sono le perdite serie più il charging. Per le linee i
  parametri sono simmetrici. Per i trasformatori (τ, φ, z_s, b_c) ↦ (1/τ, −φ, τ² z_s, b_c/τ²): impedenza e shunt vanno
  riportati attraverso il tap (senza il riporto, Y cambia di 2.2 p.u. con τ = 0.95). I decoder di arco devono essere
  direzionali: lo stesso decoder valutato nei due versi, oppure parte antisimmetrica (trasferimento) più parte simmetrica
  (perdite); mai antisimmetrici sui flussi AC.
- **S5 — permutazione dei nodi.** Già garantita dalle GNN; inclusa per completezza.
- **S6 (estensione) — invarianze di modellazione.** Fusione di rami paralleli (somma dei blocchi 2×2 di Y; con tap
  diversi il risultato non è in generale un singolo ramo con tap), bus-splitting a impedenza nulla, aggregazione di
  generatori sullo stesso bus. Non sono simmetrie di gruppo ma raffinamenti che non cambiano la fisica; la roadmap le
  chiama "modeling conventions".

**Gruppo complessivo.** Le convenzioni standard (basi p.u. per livello, θ_ref = 0, tap al lato from, posizione degli
sfasatori) sono gauge fixing del gruppo generato da (ℂ*)ᴺ locale, ℝ₊ globale (base MVA), Z₂^B e S_N.

**Prop. 5 (canonicalizzazione).** Se esiste un frame γ: U → G con γ(g·u) = g·γ(u), allora f è equivariante ⇔
f(u) = γ(u)·f̃(γ(u)⁻¹·u) con f̃ arbitraria sulla sezione Σ = {γ(u) = e}. (⇐: si sostituisce g·u e si usa
γ(g·u) = g·γ(u); ⇒: si scrive u = γ(u)·(γ(u)⁻¹u).) Ogni simmetria del catalogo ha un frame chiuso, continuo,
equivariante per permutazione e dipendente dai soli input:

| | frame |
|---|---|
| S1 | γ = θ_ref |
| S3 | γ = s(u)/s₀, con s omogenea di grado 1 (mean\|Y\|, P95 delle iniezioni di input) |
| S2 | gauge di Coulomb: ψ* = −L⁺Bᵀφ ⇒ φ_c = (I − BL⁺Bᵀ)φ (proiezione sullo spazio dei cicli) |
| S4 | orientamento fissato dai parametri del dispositivo (τ ≤ 1); le linee sono già invarianti nella rappresentazione bidirezionale |

Conseguenze: (i) se train e test stanno sulla sezione (θ_ref = 0, base fissa, orientamento fisso, φ = 0), il vincolo di
simmetria non restringe il modello sui dati: il guadagno è nullo per costruzione, in distribuzione e zero-shot a
convenzioni uguali; (ii) fuori sezione M0+Canon realizza già tutta la classe equivariante: un layer equivariante può
differire da M0+Canon solo per parametrizzazione, cioè per rappresentazione. PoC: `Canonicalize` applicato post hoc
cambia l'RMSE in distribuzione di 2·10⁻⁷ relativo. Vale a topologia fissa: nei dataset pubblici, tutti N-1, la
statistica di frame varia in distribuzione (CV di mean|Yff| 1.5–4.7%) e la canonicalizzazione post hoc dei checkpoint
pubblicati costa (VA +73% su case14, +1% su case118). Replica a topologia fissa, 5 seed: canon e M0 non si
distinguono in distribuzione (VA 0.22 ± 0.06° contro 0.29 ± 0.09°), e M0 con lo stesso percorso RNG di canon cade con
canon.

**Prop. 6 (S2).** Il contenuto gauge-invariante di una configurazione di sfasatori sono le olonomie. Su un albero
φ_c ≡ 0: gli sfasatori di una rete radiale sono puro gauge e non toccano P, Q, |V|. Senza sfasatori il vincolo S2 si
riduce a S1 sui dati; con sfasatori agisce solo lungo direzioni di gauge presenti nei dati (uno sfasatore su un ponte,
che non si installa), mentre uno sfasatore in un anello ne cambia l'olonomia.

**Diagnostica.** Errore di equivarianza per il modello f, la simmetria g e il canale c:

EE_{g,c}(f) = RMS_c[ f(g·u) − ρ(g)·f(u) ]

nell'unità del canale (per VA la differenza è ridotta a [−π, π)), riportato relativo all'RMSE in distribuzione dello
stesso canale, EE_{g,c}/RMSE_c: oltre 1, la rottura di simmetria domina l'errore su T_g. Una norma unica su
[VM, VA, PG, QG] dipenderebbe dalla scelta arbitraria delle unità, e per S1 il numeratore cresce con α in un modello che
ignora θ_ref. Per PG e QG sotto S3 la differenza va riportata nel frame originale (× k): misurata nel frame trasformato,
un modello esatto alla precisione float32 leggerebbe 10⁻² a k = 0.01. Calcolabile senza etichette su qualsiasi caso,
inclusi quelli fuori distribuzione. È un test di consistenza, non di accuratezza: un modello canonicalizzato ha EE ≡ 0
qualunque sia il suo errore. Il rifit del normalizzatore sul target è esattamente un'azione S3 con
k = base_rifit/base_sorgente su tutti gli input visibili (`vn_kv`, l'unico altro input rifittato, è mascherato nel PF):
|RMSE_rifit − RMSE_sorgente| ≤ EE_S3(k) è un'identità (540 verifiche sui checkpoint pubblicati, 0 violazioni).

**Nota sul rischio di banalizzazione (confermato).** S1–S4 sono eliminabili esattamente, e il piano lo assume invece di
combatterlo: (a) le convenzioni miste del pretraining multi-rete si canonicalizzano per campione; (b) la scala p.u. della
distribuzione è già allineata per rete dal normalizzatore, e per campione da M0+Canon, mentre ciò che resta sono gli
invarianti di S3; (c) il vantaggio della rappresentazione relativa (RPF) è una questione di rappresentazione, non di
simmetria. L'argomento di località ("le differenze d'arco sono più facili per un message passing a K hop") è falsificato
su toy: il miglior filtro lineare a K hop non è più accurato sulle differenze d'arco (scala 2×30, K = 12: errore
relativo 0.33 sugli angoli di nodo, 0.41 sulle differenze d'arco). Resta una domanda empirica (H1).

## 6. Cosa costruiamo

Backbone di riferimento: GENCO (grafo eterogeneo bus/generatori) e, come controllo semplice, un MPNN alla PowerFlowNet.

| Variante | Simmetrie esatte | Meccanismo |
|---|---|---|
| M0 | S5 (S4 sulle linee) | Baseline real-valued, angoli di nodo rispetto allo slack, normalizzatore per rete |
| M0+Canon | S1, S3, S4 (+ S2 con Coulomb) | Frame dei soli input (Prop. 5), zero parametri: **baseline di ogni confronto** |
| M0+Aug | — | Augmentation casuale: α ∈ [0, 2π), k ∈ [10⁻², 10²], orientamenti casuali, gauge locali casuali (sfasamenti virtuali sulle linee). **Loss valutata nel frame del campione** (predizioni de-aumentate, residuo fisico × k): altrimenti il peso della loss fisica varia di 10⁴ sul range e il baseline è handicappato (PoC: VA 3–9° contro 0.5° di M0). Con GNS non basta: su α ±π, k 0.1–10 l'addestramento fallisce anche nel frame del campione (5.7°) e senza loss fisica (5.0°), mentre ogni asse da solo addestra (fase 0.47°, scala 1.0°). Ipotesi da testare: GNS somma a ogni layer physics_mlp(residuo) allo stato latente, e il residuo scala come 1/k |
| M1 | S1 | Output = differenze angolari invarianti θ_f − θ_t − φ_ft; θ ricostruito per proiezione di Hodge θ = L_w⁺BᵀWδ̂ + θ_ref (non per albero ricoprente, che rompe S5 e accumula errore lungo i cammini); il residuo è la violazione KVL. Prima prova (2/10/2026, testa d'arco sugli embedding finali di GNS, dentro M0+Canon, 5 seed): peggio in distribuzione (VA 0.73° contro 0.22°, anche sulle differenze d'arco), uguale in zero-shot; confondente: gli angoli ricostruiti saltano la correzione fisica per layer di GNS, che resta sugli angoli di nodo del backbone. Con la ricostruzione dentro ogni layer (4 seed) il divario sparisce: in distribuzione VA 0.17° contro 0.22° di M0+Canon, intervalli sovrapposti, e nessuna separazione in zero-shot (case30 VM 0.011 contro 0.016, sovrapposti per 3·10⁻⁴). A questa scala M1 è alla pari con gli angoli di nodo: H1 per M1 non è sostenuta |
| M2 | S1 + S2 | Feature di nodo complesse, message passing con trasportatori U_ij = e^{jφ_ij}, non-linearità modReLU/cardioid; output V e^{jθ} relativo. **Controllo:** stessa rete con attivazione split Re/Im (non equivariante), per separare simmetria e algebra complessa |
| M3 | S3 | Feature adimensionali p_i = P_i/D_i, q_i = Q_i/D_i, W_ij = \|Y_ij\|/D_i (diretta), D_i = Σ_j \|Y_ij\|. Iniettiva modulo la scala globale perché \|Y_ij\| = \|Y_ji\| dà D_j/D_i = W_ij/W_ji; una "media locale" simmetrica non è garantita iniettiva e imporrebbe invarianza a riscalature locali che non sono simmetrie. Con input adimensionali i layer di omogeneità di grado 0 sono ridondanti |
| M4 | S4 | Decoder di arco direzionali (vedi S4); il tap come rapporto visto dal bus sorgente di ciascuna riga (τ nel verso from→to, 1/τ nel verso opposto), oppure il frame di orientamento |
| M5 | S1–S5 | Combinazione |

Tutte le varianti sono implementate come layer plug-in per il framework GridFM (PyTorch Geometric), rilasciati open
source. Il PoC implementa M0, M0+Canon (S1+S3+S4) e M0+Aug con entrambe le versioni della loss.

## 7. Piano sperimentale

### 7.1 Dati
- **Trasmissione:** dataset PF/OPF generati con gridfm-datakit su IEEE 14/30/57/118/300, PEGASE 1354/2869, ACTIVSg
  2000/10k. I checkpoint GENCO pubblicati coprono IEEE 14/30/57/118 e GOC 500/2000/10000.
- **Distribuzione:** ENGAGE (LV/MV, reti multiple; è lo split "reti non viste" usato da Okoyomon et al., quindi confronto
  diretto).
- **Audit delle convenzioni (E0, E0p):** nelle 7 reti datakit locali e nei dataset pubblici IEEE 14/30/57/118 (tutti
  gli scenari) θ_ref ≡ 0 e shift = 0, quindi S1 e S2 sono solo sintetiche. I dataset pubblici sono tutti N-1 (un ramo o
  un generatore fuori per scenario), con CV di mean|Yff| 1.5–4.7%, e contengono il 4.5–35% di scenari duplicati, sempre
  dentro uno stesso load scenario: nessun gemello tra train e test negli split pubblicati.
- **Test set trasformati T_g:** riferimento angolare diverso a slack invariato (S1), base MVA diversa (S3), rami
  riorientati, linee e trasformatori con la regola corretta (S4), sfasamenti ridistribuiti via gauge (S2). Servono per
  EE_g e come unit test (errore identico su T_g e sull'originale per i modelli canonicalizzati).
- Un sottoinsieme con sfasatori reali (PEGASE, ACTIVSg) per H3, se E0p li conferma.

### 7.2 Split
1. In distribuzione.
2. Regimi stressati (scaling di carico fuori dal range di training).
3. Reti non viste, stessa famiglia (train su un sottoinsieme di reti di trasmissione, test sulle altre).
4. Cross-dominio trasmissione → distribuzione.
5. Few-shot: fine-tuning con 0 / 10 / 100 / 1000 campioni della rete target.

Nello zero-shot il normalizzatore resta quello della sorgente, oppure si usa un frame dei soli input: rifittarlo sul
target legge Qg e il Pg dello slack, cioè etichette (PoC, M0 seed 0: il cambio di convenzione sposta le metriche fino a
2×, con segno variabile; M0+Canon ne è indipendente per costruzione). Il rifit inoltre agisce sugli input come
un'azione S3 esatta: senza canonicalizzazione la differenza di metrica tra i due protocolli è la rottura di S3 del
modello a quel k (§5, Diagnostica); con M0+Canon la scelta è irrilevante.

### 7.3 Baseline e confronti
M0+Canon come riferimento per ogni variante; M0; M0+Aug con loss nel frame del campione e budget di campioni pari; DC-PF
e fast-decoupled come riferimenti fisici (Okoyomon mostra che spesso battono le GNN OOD); GENCO pubblicato.

### 7.4 Metriche
- Accuratezza: MAE su V e θ (θ valutato come θ_f − θ_t − φ_ft, invariante di gauge, per non premiare artefatti di
  riferimento), errore sui flussi. L'errore sugli angoli di nodo è per l'81–96% un offset comune per grafo rispetto al
  REF, visto solo dai rami incidenti al REF (replica, tutti i bracci): va riportato separato dal resto e da θ_f − θ_t
  (`angle_error_split`).
- Fisica (formato GridBench): distribuzione dei residui di power balance, tassi di violazione di tensione/termici,
  frazione di casi entro tolleranza ingegneristica, code (P95/P99).
- **EE_{g,c}/RMSE_c per ciascuna simmetria e canale** (nuova).
- Efficienza dati: curve di apprendimento vs numero di campioni.
- Transfer: zero-shot e curve few-shot.

### 7.5 Tabella dei run

| ID | Modello | Training | Test | Verifica |
|---|---|---|---|---|
| R1 | M0, M0+Canon, M0+Aug, M1–M5 | in-dist, ogni rete | in-dist + T_g | EE_g; Prop. 5 (M0+Canon ≡ M0 in-dist) |
| R2 | tutti | multi-rete trasmissione | reti tenute fuori | H1, H4 |
| R3 | tutti | trasmissione | ENGAGE zero/few-shot | H1, H4 (rappresentante S3) |
| R4 | M2, M2-split, rete complessa + Coulomb | sottoinsieme con/senza sfasatori | idem | H3 |
| R5 | tutti | curve 10²–10⁵ campioni | in-dist e OOD | H1 (efficienza) |
| R6 | modelli non equivarianti | analisi trasversale | — | H2, dentro classe |
| R7 | M0, M0+Aug fase | in-dist | in-dist | meccanismo del guadagno dell'augmentation di fase (PoC: VA 0.09° contro 0.51°, non spiegato dalla simmetria perché canon ≡ M0 in-dist): compare anche su θ_f − θ_t o solo sugli angoli di nodo (instradamento del riferimento)? **Fatto:** il guadagno è uniforme su offset, resto e θ_f − θ_t (rapporti 0.42/0.48/0.41): regolarizzazione, non instradamento; in zero-shot su case30 l'augmentation di fase peggiora proprio l'offset (5.7–7.1° contro 1.5°; osservazione post hoc) |

3 seed per configurazione; report con intervalli.

## 8. Componente teorica
- **Proposizioni 1–4:** invarianza/equivarianza di Φ sotto S1–S4, con S4 nella forma corretta (scambio dei flussi, regola
  dei trasformatori), e caratterizzazione delle scelte di rappresentazione che le rompono: angoli di nodo con slack fisso
  (S1), input p.u. senza frame (S3), il tap come feature del lato from (S4 sui trasformatori; M0 addestrato, replica su
  5 seed: EE/RMSE 0.02–0.11 sui trasformatori contro ≤ 9·10⁻⁵ sulle linee, e ≥ 390 su S1 a α = π, dove l'input θ_ref
  non ha mai variato; checkpoint GENCO pubblicati: 0.05–0.37, ~3·10⁻⁴ e 300–6700).
- **Prop. 5 (canonicalizzazione)** e **Prop. 6 (olonomie; S2 su alberi e senza sfasatori)**, §5.
- **Lemma:** M2 è esattamente covariante sotto S2 per costruzione (dimostrazione standard di gauge-covarianza del message
  passing con trasportatori).
- **Elesedy & Zaidi 2021 non motiva H1:** il teorema (da verificare) assume G compatto e una distribuzione degli input
  G-invariante; qui la distribuzione è concentrata su una sezione e ℝ₊ (S3) non è compatto.

## 9. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Obiezione "basta canonicalizzare" | Confermata dalla Prop. 5 per la parte di simmetria: il piano la assume (M0+Canon baseline); il contributo è il catalogo, i frame, EE come test di pipeline e i confronti di rappresentazione |
| Instabilità del training complesso | modReLU, normalizzazione complessa, inizializzazione unitaria; confronto con M1 (real-valued, S1 esatta) come fallback |
| Tre simmetrie su quattro solo sintetiche nei dati datakit (θ_ref ≡ 0, shift = 0, S4 esatta sulle linee; confermato sui dati pubblici, E0p) | Evidenza reale solo da convenzioni diverse (ENGAGE); i T_g sintetici servono per EE_g |
| Baseline M0+Aug che non addestra sull'intero range | La loss nel frame del campione non basta (E6b), né togliere la loss fisica (E6a): range mild come baseline e M0+Canon come riferimento; testare la retroazione fisica di GNS (residuo normalizzato in ingresso a physics_mlp) |
| Frame che legge etichette (normalizzatore rifittato sul target) | Zero-shot con normalizzatore della sorgente o frame dei soli input (§7.2) |
| Tap e sfasatori complicano S4 | Regola corretta (1/τ, −φ, τ²z_s, b_c/τ²); nella rappresentazione GridFM (Y_self, Y_mutual per riga) lo scambio delle righe la realizza, e restano da trasformare solo tap e limiti angolari |
| Guadagni piccoli in distribuzione | Attesi nulli per la parte di simmetria (Prop. 5): il paper è sul transfer e sulla rappresentazione |

## 10. Deliverable, fasi, calcolo

**Deliverable:** (1) catalogo con dimostrazioni e frame; (2) libreria di frame e layer compatibile con il framework
GridFM; (3) tool per EE_g e i test set trasformati; (4) paper; (5) eventuale PR al framework GridFM (frame dei soli input
nel normalizzatore, tap indipendente dall'orientamento).

**Fasi (circa 5–6 mesi):**
- F0 (4 settimane): formalizzazione; EE_g sui checkpoint GENCO pubblici (per S1 il risultato è prevedibile, input mai
  variato: vale come test di pipeline, non come scoperta). Fatto il 2/10/2026 (`results/genco/VERDICTS.md`).
- F1 (6–8 settimane): M0+Canon, M1–M4 su reti piccole/medie, R1, R5, R7.
- F2 (8 settimane): scala e transfer, R2–R4, R6.
- F3 (4 settimane): teoria, scrittura, rilascio codice.

**Calcolo (stima grossolana):** GNN su reti ≤ 10k bus; 1–2 GPU classe A100 per 6–8 settimane complessive coprono la
griglia di run con 3 seed. Il costo dominante è M2 (feature complesse ≈ 2× memoria).

## 11. Paper e venue

Due possibili tagli, non esclusivi:
- **Paper 1 (diagnostico):** "Le convenzioni del power flow come gauge fixing: catalogo, canonicalizzazione esatta e un
  test di consistenza per surrogati neurali" — catalogo unificato con i frame, EE come unit test di pipeline, correzioni
  pratiche (tap indipendente dall'orientamento, frame dei soli input, zero-shot senza etichette del target). Venue:
  PSCC 2027 / IEEE TPWRS / EPSR.
- **Paper 2 (rappresentazione):** subordinato a R2–R4; sostenibile solo se una scelta di rappresentazione (M1, M2 contro
  M2-split, M3) batte M0+Canon oltre la variabilità tra seed. La simmetria di gauge da sola non lo giustifica
  (Prop. 5–6). Venue: NeurIPS/ICLR (track geometric DL o AI for science), oppure TPWRS se il taglio è più applicativo.

**Contributo al progetto GridFM:** frame e layer riusabili nel backbone; EE_g come check di introspezione in §4.6;
canonicalizzazione per il pretraining multi-rete in §4.4.

## 12. Collegamento con il Piano B

La loss Sobolev del Piano B va scritta in forma covariante: se f è equivariante, il suo Jacobiano soddisfa
J_f(g·u) = ρ_out(g) J_f(u) ρ_in(g)⁻¹, con ρ la parte lineare dell'azione (per S1, affine su θ, il Jacobiano è
invariante). Con M0+Canon basta scrivere la loss nel frame canonico. I due piani condividono backbone, dati e split.

## 13. Riferimenti (da verificare prima dell'uso)

- E. Okoyomon, A. Yaniv, C. Goebel, "Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids", arXiv:2509.25158, 2025. Dataset ENGAGE: doi 10.5281/zenodo.15464235.
- J. Stiasny, J. Cremer, "Residual Power Flow for Neural Solvers", arXiv:2601.09533, 2026.
- P. Dogoulis, K. Tit, M. Cordy, "KCLNet: Physics-Informed Power Flow Prediction via Constraints Projections", arXiv:2506.12902, 2025.
- "Flow-Attentional Graph Neural Networks", arXiv:2506.06127, 2025.
- "Gauge-Equivariant Graph Networks via Self-Interference Cancellation", arXiv:2511.16062, 2025.
- T. Cohen, M. Weiler, B. Kicanaoglu, M. Welling, "Gauge Equivariant Convolutional Networks and the Icosahedral CNN", ICML 2019 (arXiv:1902.04615).
- M. Favoni, A. Ipp, D. Müller, D. Schuh, "Lattice gauge equivariant convolutional neural networks", arXiv:2012.12901, 2021.
- S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio, S. Ravanbakhsh, "Equivariance with Learned Canonicalization Functions", ICML 2023.
- O. Puny et al., "Frame Averaging for Invariant and Equivariant Network Design", ICLR 2022.
- N. Dym, H. Lawrence, J. W. Siegel, "Equivariant Frames and the Impossibility of Continuous Canonicalization", ICML 2024.
- A. Puech et al., "GENCO — A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis", arXiv:2608.09921, 2026; gridfm-datakit, arXiv:2512.14658, 2025.
- A. Yaniv, C. Goebel, "Benchmarking graph neural networks for power flow prediction in distribution systems", IEEE PowerTech 2025.
- A. Varbella et al., "PowerGraph: a power grid benchmark dataset for graph neural networks", NeurIPS 2024.
- B. Donon et al., "Neural networks for power flow: Graph neural solver", EPSR 189, 2020.
- N. Lin et al., "PowerFlowNet", IJEPES 160, 2024.
- S. Dhople et al., "Reexamining the Distributed Slack Bus", IEEE TPWRS 35(6), 2020.
- B. Elesedy, S. Zaidi, "Provably strict generalisation benefit for equivariant models", ICML 2021 (da verificare; ipotesi: G compatto, distribuzione degli input G-invariante).
- GridFM Roadmap, "Shared Foundations, Better Outcomes", §4.2, §4.4, §4.6.
