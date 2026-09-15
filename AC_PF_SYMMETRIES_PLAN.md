# Piano A — Simmetrie della mappa AC power flow come bias induttivi per Grid Foundation Models

*Bozza di piano di ricerca — settembre 2026*

---

## 1. In una frase

Catalogare le trasformazioni che lasciano invariante la soluzione del power flow, dimostrare quali di esse le pipeline neurali standard rompono, misurare quanto quella rottura spiega il fallimento del transfer zero-shot, e costruire layer esattamente equivarianti che la eliminano.

## 2. Motivazione e posizionamento nella roadmap GridFM

La roadmap identifica il transfer zero-shot a reti non viste come il collo di bottiglia che separa un "shared grid model" (livello 1) da un "transferable GridFM" (livello 2), e riporta che le reti a valori complessi migliorano la predizione OOD degli angoli di circa due ordini di grandezza *attribuendo il guadagno al bias induttivo e non alla capacità* (§4.2). Quel risultato è un caso particolare di un principio più generale — la mappa AC-PF ha simmetrie esatte — che nessuno ha ancora enumerato e sfruttato sistematicamente.

Il piano copre:
- **§4.2 (architetture)**: layer equivarianti come alternativa strutturata all'augmentation.
- **§4.4 (pretraining e transfer)**: le simmetrie sono augmentation esatte e gratuite per il pretraining multi-rete, e spiegano perché le convenzioni di modellazione (base p.u., slack, orientamento) ostacolano il transfer.
- **§4.6 (introspezione)**: l'*errore di equivarianza* di un modello addestrato è un test di consistenza fisica calcolabile senza etichette.

## 3. Stato dell'arte e gap

| Lavoro | Cosa fa | Cosa manca |
|---|---|---|
| Okoyomon, Yaniv, Goebel 2025 (arXiv:2509.25158) | Ablazione controllata di tre bias (PF-loss, reti complesse, riformulazione residuale) su ENGAGE; reti complesse battono DC-PF di un ordine di grandezza sugli angoli OOD | Un solo bias legato alla simmetria (fase); nessun catalogo; nessuna misura dell'equivarianza |
| Stiasny & Cremer 2026, *Residual Power Flow* (arXiv:2601.09533) | Angoli di ramo invece di angoli di bus: elimina il bus di riferimento; mostra che la relazione angolo-potenza diventa quasi lineare e più facile da apprendere | 9 bus, MLP; la KVL diventa un residuo; gli autori stessi indicano architetture graph-based come passo successivo |
| Dogoulis et al. 2025, KCLNet (arXiv:2506.12902); Flow-Attentional GNN (arXiv:2506.06127) | Vincoli di conservazione (KCL) imposti strutturalmente | Non trattano simmetrie di fase, scala, orientamento |
| GNN gauge-equivarianti U(1) (arXiv:2511.16062); Cohen et al. 2019; Favoni et al. 2021 (L-CNN) | Teoria e layer per equivarianza di gauge locale su grafi e reticoli | Mai applicati a reti elettriche |
| Puech et al. 2026, GENCO (arXiv:2608.09921) | Backbone eterogeneo PF/OPF/SE, decoder fisici, correzione iterativa; dataset multi-rete | Rappresentazione real-valued, angoli di nodo rispetto allo slack, input p.u. a base fissa: rompe le simmetrie S1–S4 sotto |

**Gap:** non esiste (i) un catalogo formale delle simmetrie della mappa AC-PF, (ii) una diagnostica dell'errore di equivarianza per modelli addestrati, (iii) un confronto controllato "equivarianza esatta vs augmentation" su più reti, task e regimi.

## 4. Domande di ricerca e ipotesi

**RQ1.** Quali trasformazioni lasciano invariante/equivariante la soluzione AC-PF, e quali sono rotte dalle scelte di rappresentazione standard?

**RQ2.** L'errore di equivarianza dei modelli addestrati predice il loro errore su reti non viste?

**RQ3.** Architetture esattamente equivarianti battono l'augmentation in efficienza dati e in generalizzazione OOD?

- **H1.** I modelli equivarianti eguagliano l'augmentation con meno dati e la battono zero-shot su reti non viste; il guadagno è concentrato su angoli e flussi (S1/S2) e sul transfer trasmissione→distribuzione (S3).
- **H2.** Esiste una correlazione forte (ρ > 0,7 tra modelli e seed) tra errore di equivarianza e errore OOD.
- **H3.** Il gauge locale (S2) conta solo in presenza di sfasatori; fase globale (S1) e scala (S3) contano ovunque.
- **H4 (negativa, da controllare).** Fissare le convenzioni a priori (100 MVA, θ_slack = 0) *non* recupera i benefici, perché nel pretraining multi-rete le convenzioni si mescolano e perché la rappresentazione relativa generalizza meglio anche a convenzione fissa.

## 5. Formalizzazione

Rete G = (N, B), matrice di ammettenza Y (con tap r_ij e sfasamenti φ_ij sui trasformatori), setpoint u (P, Q ai bus PQ; P, V ai PV; V, θ allo slack). Mappa soluzione Φ: (G, Y, u) ↦ x = (V, θ).

Simmetrie (ciascuna diventa una proposizione con dimostrazione di due righe nel paper):

- **S1 — U(1) globale.** g_α: θ_i ↦ θ_i + α ∀i. Tutti i flussi e le iniezioni sono invarianti; Φ è equivariante. Con θ_slack fissato a 0 la simmetria riappare come libertà di scelta del bus di riferimento r: l'output fisicamente significativo è θ_i − θ_r, oppure le differenze θ_i − θ_j sugli archi.
- **S2 — gauge U(1) locale.** Per ogni bus j e φ ∈ ℝ: θ_j ↦ θ_j + φ e, per ogni ramo incidente, φ_ij ↦ φ_ij ∓ φ (segno per orientamento). La fisica è invariante. È esatta sulla famiglia di reti parametrizzata dagli sfasamenti: gli sfasatori sono una *connessione* sugli archi, e il message passing complesso con trasportatori U_ij = e^{jφ_ij} è gauge-covariante. S1 è il caso φ costante.
- **S3 — scala (analisi dimensionale).** Cambio di base MVA k > 0: (P, Q, Y, shunt) ↦ (P, Q, Y, shunt)/k con V_pu invariato. Φ è invariante. Estensione: cambio di base di tensione per livello, che riscala V e Y e lascia S invariata; i rapporti di tap ne assorbono l'effetto attraverso i trasformatori.
- **S4 — orientamento dei rami (Z₂ per arco).** Invertire from/to nega flusso e differenza angolare; per le linee i parametri sono simmetrici, per i trasformatori il lato tap si scambia (r ↦ 1/r, φ ↦ −φ nella convenzione Π). I decoder sugli archi devono essere antisimmetrici sulle linee e direzionali solo sui trasformatori.
- **S5 — permutazione dei nodi.** Già garantita dalle GNN; inclusa per completezza.
- **S6 (estensione) — invarianze di modellazione.** Fusione di rami paralleli (Y₁ + Y₂), bus-splitting a impedenza nulla, aggregazione di generatori sullo stesso bus. Non sono simmetrie di gruppo ma raffinamenti che non cambiano la fisica; la roadmap le chiama "modeling conventions".

**Diagnostica.** Errore di equivarianza per il modello f e la simmetria g:

EE_g(f) = E_{u, g} ‖ f(g·u) − g·f(u) ‖ / ‖ f(u) ‖

calcolabile senza etichette su qualsiasi caso, incluso quelli fuori distribuzione.

**Nota sul rischio di banalizzazione.** S1 e S3 sono "eliminabili" fissando convenzioni. Il piano le tratta comunque perché: (a) dataset di reti diverse usano convenzioni diverse e il pretraining multi-rete le mescola; (b) la distribuzione lavora a scale p.u. incompatibili con la trasmissione, quindi S3 è la chiave per un backbone unico; (c) RPF mostra che la rappresentazione relativa (angoli di ramo) è più facile da apprendere anche a convenzione fissa. H4 testa esplicitamente questa obiezione.

## 6. Cosa costruiamo

Backbone di riferimento: GENCO (grafo eterogeneo bus/generatori) e, come controllo semplice, un MPNN alla PowerFlowNet.

| Variante | Simmetrie esatte | Meccanismo |
|---|---|---|
| M0 | S5 | Baseline real-valued, angoli di nodo rispetto allo slack, input p.u. |
| M0+Aug | — | Augmentation casuale: α ∈ [0, 2π), bus di riferimento casuale, k ∈ [10⁻², 10²], orientamenti casuali, sfasamenti riassorbiti |
| M1 | S1 | Output = differenze angolari sugli archi; ricostruzione θ per integrazione su albero ricoprente; KVL residuo sui cicli |
| M2 | S1 + S2 | Feature di nodo complesse, message passing con trasportatori U_ij = e^{jφ_ij}, non-linearità modReLU/cardioid; output V e^{jθ} relativo |
| M3 | S3 | Feature adimensionali: iniezioni normalizzate per Σ_j |Y_ij| (o per la potenza di corto circuito), ammettenze normalizzate per la loro media locale; layer con omogeneità di grado 0 |
| M4 | S4 | Decoder di arco antisimmetrici f(h_i, h_j) = −f(h_j, h_i) sulle linee; feature direzionali solo sui trasformatori |
| M5 | S1–S5 | Combinazione |

Tutte le varianti sono implementate come layer plug-in per il framework GridFM (PyTorch Geometric), rilasciati open source.

## 7. Piano sperimentale

### 7.1 Dati
- **Trasmissione:** dataset PF/OPF generati con gridfm-datakit su IEEE 14/30/57/118/300, PEGASE 1354/2869, ACTIVSg 2000/10k (stessi casi dei rilasci GENCO per confrontabilità diretta).
- **Distribuzione:** ENGAGE (LV/MV, reti multiple; è lo split "reti non viste" usato da Okoyomon et al., quindi confronto diretto).
- **Test set trasformati T_g:** gli stessi casi con base MVA diversa (S3), slack/riferimento diverso (S1), rami riorientati (S4), sfasamenti ridistribuiti via gauge (S2). Servono sia per EE_g sia per verificare che i modelli equivarianti abbiano errore identico su T_g e sull'originale.
- Un sottoinsieme con sfasatori reali (PEGASE, ACTIVSg) per testare H3.

### 7.2 Split
1. In distribuzione.
2. Regimi stressati (scaling di carico fuori dal range di training).
3. Reti non viste, stessa famiglia (train su un sottoinsieme di reti di trasmissione, test sulle altre).
4. Cross-dominio trasmissione → distribuzione (qui S3 è decisiva).
5. Few-shot: fine-tuning con 0 / 10 / 100 / 1000 campioni della rete target.

### 7.3 Baseline e confronti
M0, M0+Aug (con budget di campioni pari), DC-PF e fast-decoupled come riferimenti fisici (Okoyomon mostra che spesso battono le GNN OOD), GENCO pubblicato.

### 7.4 Metriche
- Accuratezza: MAE su V e θ (θ valutato come differenze sugli archi, per non premiare artefatti di riferimento), errore sui flussi.
- Fisica (formato GridBench): distribuzione dei residui di power balance, tassi di violazione di tensione/termici, frazione di casi entro tolleranza ingegneristica, code (P95/P99).
- **EE_g per ciascuna simmetria** (nuova).
- Efficienza dati: curve di apprendimento vs numero di campioni.
- Transfer: zero-shot e curve few-shot.

### 7.5 Tabella dei run

| ID | Modello | Training | Test | Verifica |
|---|---|---|---|---|
| R1 | M0, M0+Aug, M1–M5 | in-dist, ogni rete | in-dist + T_g | EE_g, H4 |
| R2 | tutti | multi-rete trasmissione | reti tenute fuori | H1, H2 |
| R3 | tutti | trasmissione | ENGAGE zero/few-shot | H1 (S3) |
| R4 | M0, M2 | sottoinsieme con/senza sfasatori | idem | H3 |
| R5 | tutti | curve 10²–10⁵ campioni | in-dist e OOD | H1 (efficienza) |
| R6 | tutti | analisi trasversale | — | correlazione EE_g ↔ errore OOD (H2) |

3 seed per configurazione; report con intervalli.

## 8. Componente teorica
- Proposizioni 1–4: invarianza/equivarianza di Φ sotto S1–S4, con caratterizzazione di quali scelte di rappresentazione le rompono (angoli di nodo con slack fisso rompono S1 rispetto al riferimento; input p.u. senza normalizzazione rompono S3; feature di arco direzionali rompono S4).
- Lemma: M2 è esattamente covariante sotto S2 per costruzione (dimostrazione standard di gauge-covarianza del message passing con trasportatori).
- Collegamento ai risultati noti sul beneficio di generalizzazione dei modelli equivarianti (da verificare: Elesedy & Zaidi 2021) per motivare H1.

## 9. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Obiezione "basta fissare le convenzioni" | H4 come esperimento esplicito; enfasi su multi-rete e cross-dominio |
| Instabilità del training complesso | modReLU, normalizzazione complessa, inizializzazione unitaria; confronto con M1 (real-valued, S1 esatta) come fallback |
| Tutti i dataset a 100 MVA rendono S3 sintetica | Il test reale è trasmissione → distribuzione (ENGAGE); i T_g sintetici servono solo per EE_g |
| Tap e sfasatori complicano S4 | Trattamento esplicito: antisimmetria solo sulle linee, feature direzionali sui trasformatori |
| Guadagni piccoli in distribuzione | È atteso e coerente con H1: il paper è sul transfer, non sull'interpolazione |

## 10. Deliverable, fasi, calcolo

**Deliverable:** (1) catalogo con dimostrazioni; (2) libreria di layer equivarianti compatibile con il framework GridFM; (3) tool per EE_g e i test set trasformati; (4) paper; (5) eventuale PR al framework GridFM.

**Fasi (circa 5–6 mesi):**
- F0 (4 settimane): formalizzazione; calcolo di EE_g sui checkpoint GENCO pubblici (risultato preliminare a costo quasi nullo).
- F1 (6–8 settimane): M1–M4 su reti piccole/medie, R1 e R5.
- F2 (8 settimane): scala e transfer, R2–R4, R6.
- F3 (4 settimane): teoria, scrittura, rilascio codice.

**Calcolo (stima grossolana):** GNN su reti ≤ 10k bus; 1–2 GPU classe A100 per 6–8 settimane complessive coprono la griglia di run con 3 seed. Il costo dominante è M2 (feature complesse ≈ 2× memoria).

## 11. Paper e venue

Due possibili tagli, non esclusivi:
- **Paper 1 (diagnostico):** "Le simmetrie del power flow AC e come i surrogati neurali le rompono" — catalogo, EE_g, correlazione con il transfer. Venue: PSCC 2027 / IEEE TPWRS / EPSR.
- **Paper 2 (architetturale):** GNN complessa gauge-equivariante per reti elettriche, con la teoria di gauge U(1) come cornice unificante. Venue: NeurIPS/ICLR (track geometric DL o AI for science), oppure TPWRS se il taglio è più applicativo.

**Contributo al progetto GridFM:** layer riusabili nel backbone; EE_g come check di introspezione in §4.6; augmentation esatte per il pretraining multi-rete in §4.4.

## 12. Collegamento con il Piano B

La loss Sobolev del Piano B va scritta in forma covariante: se f è equivariante, il suo Jacobiano soddisfa J_f(g·u) = ρ_out(g) J_f(u) ρ_in(g)⁻¹. Un modello equivariante ha meno gradi di libertà spuri da correggere con le sensitività; i due piani condividono backbone, dati e split.

## 13. Riferimenti (da verificare prima dell'uso)

- E. Okoyomon, A. Yaniv, C. Goebel, "Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids", arXiv:2509.25158, 2025. Dataset ENGAGE: doi 10.5281/zenodo.15464235.
- J. Stiasny, J. Cremer, "Residual Power Flow for Neural Solvers", arXiv:2601.09533, 2026.
- P. Dogoulis, K. Tit, M. Cordy, "KCLNet: Physics-Informed Power Flow Prediction via Constraints Projections", arXiv:2506.12902, 2025.
- "Flow-Attentional Graph Neural Networks", arXiv:2506.06127, 2025.
- "Gauge-Equivariant Graph Networks via Self-Interference Cancellation", arXiv:2511.16062, 2025.
- T. Cohen, M. Weiler, B. Kicanaoglu, M. Welling, "Gauge Equivariant Convolutional Networks and the Icosahedral CNN", ICML 2019 (arXiv:1902.04615).
- M. Favoni, A. Ipp, D. Müller, D. Schuh, "Lattice gauge equivariant convolutional neural networks", arXiv:2012.12901, 2021.
- A. Puech et al., "GENCO — A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis", arXiv:2608.09921, 2026; gridfm-datakit, arXiv:2512.14658, 2025.
- A. Yaniv, C. Goebel, "Benchmarking graph neural networks for power flow prediction in distribution systems", IEEE PowerTech 2025.
- A. Varbella et al., "PowerGraph: a power grid benchmark dataset for graph neural networks", NeurIPS 2024.
- B. Donon et al., "Neural networks for power flow: Graph neural solver", EPSR 189, 2020.
- N. Lin et al., "PowerFlowNet", IJEPES 160, 2024.
- S. Dhople et al., "Reexamining the Distributed Slack Bus", IEEE TPWRS 35(6), 2020.
- B. Elesedy, S. Zaidi, "Provably strict generalisation benefit for equivariant models", ICML 2021 (da verificare).
- GridFM Roadmap, "Shared Foundations, Better Outcomes", §4.2, §4.4, §4.6.
