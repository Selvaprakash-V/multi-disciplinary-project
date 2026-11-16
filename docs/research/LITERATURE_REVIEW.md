# Literature Review: Acoustic-Emission Monitoring of PV Micro-Cracks with Localization and ML Source Classification

**Date:** 2026-11-15
**Scope:** (A) the PV micro-crack problem and current detection methods; (B) passive acoustic emission (AE) sensing with PZT sensors, TDOA source localization, and machine-learning classification of AE sources (crack vs. hail, rain, wind rattle, thermal clicks).
**Sources searched:** Web search across IEEE Xplore, ScienceDirect, MDPI, Springer, arXiv, ResearchGate and institutional repositories, plus semantic paper search over Semantic Scholar, OpenAlex and PubMed (via Consensus).
**Limitations:** I had no direct subscription access to IEEE Xplore or Scopus, and the semantic search returned only 3 papers per query. Chinese, Korean and Japanese-language literature was only partly covered. Full texts were not read for every paper; figures quoted below come from abstracts and publisher summaries. Section 8 lists the searches still worth running.

---

## 1. Executive summary

1. **The problem is well documented.** Micro-cracks barely reduce power on day one, but they grow under ageing and thermal or mechanical cycling and can disconnect parts of a cell. Köntges et al. found that losing less than 8% of a cell's active area causes essentially no module power loss. Between about 12% and 50% inactive area, power loss rises roughly linearly, up to the output of a whole bypass-diode substring [R1].
2. **Current inspection is a periodic snapshot, not continuous monitoring.** Electroluminescence (EL) imaging is the reference method, and CNN classifiers on public EL datasets now reach about 88–98% accuracy [R10–R13]. Field EL still needs darkness or special daylight setups (drone daylight EL [R14], outdoor PL [R15], SWIR drone imaging [R16]), so it only gives occasional snapshots.
3. **Passive AE on rigid c-Si PV modules is almost unstudied. This is the central research gap.** I found no peer-reviewed study of continuous, in-field passive AE monitoring of glass/c-Si modules, and none on TDOA localization or ML source classification for PV. The nearest work is:
   - AE on **flexible** solar cells under tensile load. AE flagged damage at about 0.8% strain, before electrical leakage rose at about 1.2% [R20, R21].
   - AE on **silicon dies**, separating the first crack from final failure [R22].
   - **Active** ultrasonic methods on PV: Lamb waves [R17, R18], air-coupled ultrasound [R19], electromechanical impedance [R23], and resonance ultrasonic vibration for wafers [R24].
4. **The building blocks are mature in other fields and transfer well:**
   - Localization: TDOA/triangulation in anisotropic plates [R30, R31], Delta-T mapping with an AIC onset picker (about 3 mm error vs. 32.6 mm for plain TOA on a complex aluminium part) [R32, R33], and single-sensor deep-learning localization that uses edge reflections [R34–R36].
   - Classification: CNNs on wavelet scalograms [R40] and multi-branch CNNs that separate **noise vs. impact vs. crack propagation** in rail AE [R41]. That rail setup is the closest analogue to separating hail from cracks.
   - Sim-to-real: transfer learning and domain adaptation from FE-simulated AE to real data [R45–R48], and TinyML AE classifiers on microcontrollers [R50, R51].
5. **Implication:** AcoustiSolar's approach (PZT corner sensors, TDOA, ML source classification) is new for PV. Its credibility will depend on **physical validation with EL ground truth**, which the literature does not yet supply. The main technical risks are:
   - signal attenuation and dispersion in the glass/EVA/cell/backsheet stack, including the EVA's temperature-dependent stiffness;
   - hail impacts that *also cause* cracks, so both happen in the same event;
   - pre-existing cracks that close and reopen without new fracture energy [R5], which AE may never hear.

---

## 2. The real-world problem: what the literature establishes

### 2.1 Origins and distribution of cracks
- Cracks start during manufacturing (soldering and stringing, lamination), transport, installation, and in the field (snow and wind load, hail, thermal cycling). Reviews covering this: Papargyri et al. 2020 [R6], Ennemri et al. 2019 [R7], Abdelhamid, Singh & Omar 2014 (*IEEE J. Photovoltaics*) [R8], and Goudelis, Lazaridis & Dhimish 2022 [R9].
- Mechanical load tests (IEC 61215) produce characteristic spatial and orientation patterns of cracks [R3]. Loading a module while imaging it with EL or measuring I-V shows how cracks form and open under stress [R4].
- Interconnects under thermal cycling: cracks often begin at solder joints and at busbar ends. Thermal-expansion mismatch drives cracking in ECA-bonded shingled cells [R25].

### 2.2 Why micro-cracks matter (power-loss mechanics)
- **Köntges et al. 2011** (*SOLMAT*, 323 citations) [R1] and the earlier EU PVSEC paper [R2]:
  - The immediate power loss from micro-cracks is small.
  - After artificial ageing, the risk becomes large.
  - The largest inactive area within a bypass-protected double string sets the module's power loss.
  - Under 8% inactive cell area: no practical loss. From about 12% to 50%: roughly linear loss.
- **Paggi et al. 2014** (*Scientific Reports*) [R5]:
  - The encapsulant and residual stresses make cracks close when load is removed, so cracked regions can **regain electrical conductivity**.
  - Under cyclic bending, cracks **degrade by fatigue**.
  - So damage keeps changing under vibration and thermal cycling. Some of those changes (crack growth) should emit AE; others (crack opening and closing) may emit little or none.
- Testing labs (PVEL's Mechanical Stress Sequence) report that mechanical loading **creates** cracks, while later thermal cycling and humidity-freeze **turn them into power loss** [R26]. A continuous monitor could catch the creation event that periodic testing misses.

### 2.3 Current detection methods and their limits

| Method | What the literature shows | Limits relevant to continuous monitoring |
|---|---|---|
| EL imaging + deep learning | ELPV dataset: CNN 88.42% vs. SVM 82.44% (Deitsch 2019) [R10]; later CNNs 95–98% [R11, R12]. PVEL-AD: 36,543 EL images with box labels (Su et al., *IEEE TII* 2022) [R13] | Needs forward bias and darkness; manual or periodic |
| Drone daylight EL / outdoor PL | Drone EL at 120 fps in daylight (*IEEE JPV* 2020) [R14]; sunlight-excited PL (Bhoopathy 2018) [R15]; PL can miss wide dark regions that EL shows | Still campaign-based; needs string modulation hardware |
| SWIR / motion-enhanced drone imaging | DTU 2026: finds low-energy glass cracks that RGB and IR miss [R16] | Targets glass cracks, not cell cracks; periodic |
| IR thermography | Finds hot spots after damage has developed | Late indicator |
| Active Lamb waves (ultrasonic) | High-order modes are more sensitive to front-glass cracks; energy-spectral-density threshold (Silitonga/Meraghni 2023–24) [R17, R18] | Active scanning; mostly thin-film glass; lab |
| Air-coupled ultrasound | Internal crack detection in PV panels; best Lamb frequency about 200 kHz [R19] | Lab, scanning |
| Electromechanical impedance (EMI) | PZT-based RMSD index tracks crack location and depth (simulation) [R23] | Simulation only; local sensitivity |
| Resonance Ultrasonic Vibration (RUV) | In-line wafer and cell crack screening [R24] | Wafers and cells only, not laminated modules |

**Takeaway:** every established method is periodic or active. None "listens" continuously, which is the niche AcoustiSolar targets.

---

## 3. Passive AE applied to solar cells and silicon: the evidence that exists

| Study | Finding | Relevance |
|---|---|---|
| Tazawa et al. 2013 (JSME Kanto) [R21] | Flexible solar cells under tension: cumulative AE energy rose at about **0.8% strain**, dark leakage current at about **1.2%**. AE detected damage **before** electrical degradation | Direct evidence that AE gives early warning in PV devices |
| Takabayashi et al. 2022 (JSME) [R20] | AE + lock-in thermography under biaxial load; crack patterns depend on load state | Same group; confirms AE sensitivity in cells |
| AE ball-breaker test on silicon die (*J. Electron. Mater.* 2015) [R22] | AE separates **initial cracking** from final bulk failure in silicon | Silicon microcracks emit detectable AE |
| Glass AE (Challenging Glass conference) [R27] | AE used to locate the initial crack in monolithic and laminated glass panes | Glass-fracture AE can be localized |
| Wafer patent (US 8,023,110 family) [R28] | Thermally induced expansion releases strain energy at cracks as characteristic acoustic waves | Early idea for acoustic crack sensing in wafers |
| Lamb attenuation in solar-cell laminates (SHM 2025) [R29] | FE study of material damping on Lamb waves in a solar-cell laminate (space arrays) | Quantifies the propagation-loss risk |

**Not found (the gap):**
- Field or chamber passive AE on framed glass/c-Si modules.
- AE paired with EL ground truth on modules.
- AE source localization on PV modules.
- ML that separates crack AE from hail, rain, wind or thermal events on PV.
- Long-term durability of bonded PZTs on modules.

---

## 4. Localization ("reverse GPS"): state of the art

### 4.1 Classical TDOA / triangulation
- **Kundu 2014, "Acoustic source localization"** (*Ultrasonics* 54(1):25–38) [R30] is the standard review. It compares methods that need direction-dependent velocity profiles with methods that do not, for isotropic and anisotropic plates.
- Sensor-cluster methods find the wavefront direction from three closely spaced sensors and need **no material properties and no straight-ray assumption** (Park, Packo & Kundu 2016; Nakatani, Kundu & Takeda 2014) [R31].
- A 2024 review (*Applied Sciences*) notes that simple time-difference methods assume a known, constant, isotropic velocity [R31]. A multilayer PV laminate violates that assumption: the waves are dispersive and multimodal.

### 4.2 Calibration-based mapping
- **Delta-T mapping** (Baxter et al. 2007, Cardiff) builds a calibrated grid of arrival-time differences from artificial sources (Hsu-Nielsen pencil-lead breaks, ASTM E976). It needs no wave speed and no geometric model [R32].
- **AIC-Delta-T** (Pearson et al. 2016, *SHM*): on a complex aluminium specimen, mean location error was **32.6 mm (TOA) → 5.8 mm (Delta-T) → 3 mm (AIC-Delta-T)** [R32].
- **Automated Delta-T** (Al-Jumaili et al. 2016, *MSSP*) cut processing time from about 8 h to about 18 s using clustering. It has been used on an A320 wing section (Grigg et al. 2021) [R33].
- **AIC onset picking** (Maeda 1985) is the standard arrival picker in this line of work [R32].

### 4.3 Data-driven / single-sensor localization
- **Ebrahimkhanlou & Salamone:** single-sensor localization that uses multipath edge reflections. The analytical version appeared in *Ultrasonics* 2017 [R34]; the deep-learning versions use stacked autoencoders (*Aerospace* 2018) [R35]. A *MSSP* 2019 follow-up (with Dubuc) covers rivets, doublers and stringers and outperforms triangulation, SVM and shallow networks [R36]. A PV module's aluminium frame creates strong edge reflections, and this approach turns that from a problem into a signal.
- **TinyML localization:** on-device PZT impact localization on PMMA plates in under 400 ms with over 90% accuracy (random forest / shallow NN) [R51]; microcontroller TOA networks (*Computers* 2023) [R50].

### 4.4 Physics specific to PV laminates
- **Interlayer damping hits the antisymmetric (flexural) modes hardest.** In laminated glass, the polymer interlayer damps A-modes more than S-modes, so detection should favour the **symmetric modes** [R37]. EVA should behave similarly to the PVB studied there, but this needs to be verified.
- Attenuation comes from geometric spreading, material damping, leakage into neighbouring media and dispersion. Damage *raises* attenuation [R38].
- Air-coupled ultrasound work found about **200 kHz** optimal for Lamb-wave crack sensitivity in PV panels [R19]. This is a useful anchor for choosing the sensor band.

---

## 5. ML source classification ("what caused it?")

### 5.1 Reviews and established pipelines
- **Ciaburro & Iannace 2022** (*Applied Sciences*, 160 citations) [R39] review ML for AE detection, localization, fracture characterization and failure-mode prediction, especially in noisy settings.
- **Traditional pipeline:** hand-crafted AE features (amplitude, rise time, counts, duration, energy, RA value, peak and centroid frequency), followed by unsupervised clustering (PCA + k-means, Kohonen self-organizing maps) or supervised trees and SVMs [R42].

### 5.2 Deep learning on time-frequency representations
- **Sikdar et al. 2021** (*Composites Part B*, 194 citations) [R40]: CNN on **CWT scalograms** from a PZT network to classify damage-source regions. Using only the peak-signal portion improved accuracy.
- **Li et al. 2020** (*SHM*, 117 citations) [R41]: **synchrosqueezed wavelet transform + two-branch CNN** to classify **operational noise vs. impact vs. crack propagation** in rails, trained on field and lab data, with transfer learning and Bayesian hyperparameter tuning. It handles cases where impact and crack signals appear together. **This is the closest published analogue to separating hail impacts from cracks.**
- **Krishna et al. 2024** (*SHM*) [R43]: a convolutional autoencoder, then k-means in latent space to *generate labels*, then a CNN classifier: 96.9% / 96.4% on pure modes and 92.5% / 91.3% on mixed modes. This answers the question of how to label real AE without ground truth.

### 5.3 Noise rejection in field AE
- Field AE noise comes from friction, reflections, rain, dust and wind. Earlier work used RMS criteria to separate friction from cracking; newer work uses RNNs and autoencoders for denoising and real-time noise/AE separation [R44].
- Rain classification exists mainly in passive acoustics (ecoacoustics, underwater): acoustic indices + MFCCs reach about 93% for binary heavy-rain detection [R44]. **No AE-sensor study of rain discrimination on panels was found.**

### 5.4 Sim-to-real: training on simulated AE
- **FE-generated AE + unsupervised domain adaptation** (manifold embedded distribution alignment) for zonal localization on steel plates, with **no real training signals** (*MSSP* 2023) [R45].
- **Domain adaptation** between simulated and experimental Hsu-Nielsen sources: about 20% gain on imbalanced data (*Eng. Res. Express* 2024) [R46].
- **Pretrain on FEM impulses, fine-tune on few experiments** (ASNT) [R47]. Multi-fidelity GAN/Inception AE localization (MSU dissertation) [R47]. Multi-fidelity convolutional-autoencoder transfer learning for guided waves (arXiv 2026) [R48].
- **The sim-to-real gap is real:** simulation-trained networks do markedly worse on real data unless the domains are aligned [R49].

### 5.5 Edge deployment
- **TinyML AE classification** on an Arduino Nano 33 BLE: over 99% on a public 5-sensor concrete AE dataset (arXiv 2411.17733) [R50]. Low-cost AE sensor nodes for long-term monitoring exist [R50].

### 5.6 Discriminating features suggested by the physics (to be validated)
These are working hypotheses drawn from general AE physics and the sources above. None has been tested on PV modules.

| Source | Expected signature |
|---|---|
| Silicon or glass fracture | Very fast rise time, broadband up to hundreds of kHz, S0 content, single burst |
| Hail impact | Large, longer, lower-frequency flexural (A0) response with ringing. Global module vibration around 10–15 Hz modes plus a short high-frequency contact transient [R52]. **May be followed within milliseconds by a crack burst** |
| Rain | Dense trains of low-energy impacts; statistics matter more than any single waveform |
| Wind rattle / clamp friction | Repeated, low-frequency-dominant events located at the **frame or clamps**. Location alone is a strong feature |
| Thermal "clicks" (stick-slip, frame/laminate expansion) | Correlated with the rate of temperature change and with sunrise/sunset; located at edges or the frame; burst shape similar to cracks |

**Design implication:** fuse waveform features with **context features**: estimated location (cell interior vs. frame or clamp), co-occurrence with an impact, temperature rate dT/dt, wind speed, rain sensor. Also use **open-set / novelty detection**, because field noise sources are never fully represented in training data.

---

## 6. Gap analysis → research opportunities for AcoustiSolar

| # | Gap in the literature | Suggested AcoustiSolar contribution |
|---|---|---|
| G1 | No passive AE study on framed glass/c-Si modules | First lab demonstration: PZT corners + EL before and after each event |
| G2 | No measured AE propagation data for PV laminates (velocity, dispersion, attenuation vs. temperature) | Characterize S0/A0 velocity and attenuation with pencil-lead breaks from −20 to 85 °C, across the EVA glass transition |
| G3 | No localization benchmark on PV | Compare model-based TDOA with AIC-Delta-T, cluster TDOA and single-sensor DL; report error in mm and at cell level |
| G4 | No crack-vs-environment AE dataset | Public dataset: controlled cracks (bend / static load per IEC 61215 / dynamic load per IEC TS 62782), ice-ball hail (IEC 61215 hail test), simulated rain, wind shaker, thermal cycling chamber |
| G5 | No link between AE energy and crack severity or inactive area | Regress AE energy / count against EL-measured inactive area, using Köntges' 8% / 12–50% thresholds as severity classes |
| G6 | No sim-to-real study for PV AE | Use the existing physics + DAQ-twin to pretrain, then domain adaptation / fine-tuning on the lab dataset |
| G7 | No study of long-term PZT bonding on modules (UV, damp heat, 25 years) | Sensor durability under IEC damp-heat / UV sequences |

---

## 7. Recommendations mapped to this repository

The repo already has Lamb-wave/SAFE physics, a DAQ twin, AIC and STA/LTA pickers, a TDOA localizer, and gradient-boosting + 1D-ResNet classifiers trained on synthetic data. The literature suggests:

1. **Localization (`packages/dsp`):** add an **AIC-Delta-T mapping** localizer as a calibration-based baseline next to model-based TDOA. It is robust to unknown, temperature-dependent velocities in the laminate [R32]. Model **frame edge reflections** explicitly, or use them as in [R34–R36].
2. **Mode / band selection (`packages/physics`, `daqtwin`):** prefer S0-dominated arrivals for timing because EVA damps A-modes [R37]. Check the sensor band against the roughly 200 kHz sensitivity anchor [R19].
3. **Classifier (`packages/ml`):**
   - add a 2D time-frequency branch (CWT or SWT scalogram CNN) [R40, R41];
   - add an explicit **"impact + crack" compound class** or a multi-label output [R41];
   - add context features (location region, dT/dt, impact co-occurrence);
   - add an open-set / out-of-distribution rejection head;
   - use domain adaptation for sim-to-real [R45, R46].
4. **Validation (`validation/`):** the V&V suite is simulation-only. The literature cannot yet supply real PV AE data, so plan a **physical validation campaign with EL ground truth** (gap G1/G4). Until then, report simulated results as "simulation-verified, not field-validated."
5. **Positioning:** present AE as **complementary** to EL. It catches new fracture *events* in real time, but it cannot see cracks that existed before installation and stay dormant (crack closure and reopening, Paggi [R5]). A baseline EL at commissioning plus continuous AE is the defensible claim.

---

## 8. Searches still worth running
- IEEE Xplore (PVSC, *IEEE JPV*) and EU PVSEC proceedings with institutional access: "acoustic emission" AND (module OR "solar cell").
- CNKI (Chinese): 声发射 光伏 组件 隐裂. Several Chinese groups work on PV hidden cracks.
- J-STAGE: the full Tazawa / Takabayashi papers for AE parameters (sensor band, thresholds).
- IEA PVPS Task 13 *Review of Failures of Photovoltaic Modules* (2014) and later Task 13 reports on cell cracks (not retrieved in this session).
- Patent search (Espacenet / Google Patents) for "acoustic emission" + "photovoltaic" monitoring systems. This matters for freedom-to-operate before any commercial claims.

---

## References
(Numbered as cited above. All links were found in this session's searches. Figures come from abstracts or publisher summaries unless stated otherwise.)

**PV cracks: impact and reviews**
- [R1] Köntges M. et al. (2011). *The risk of power loss in crystalline silicon based photovoltaic modules due to micro-cracks.* Solar Energy Materials & Solar Cells. https://consensus.app/papers/details/w:68adc43e-3740-4c61-b8be-6dcbabda9047/
- [R2] Köntges M. et al. (2010). *Quantifying the Risk of Power Loss in PV Modules Due to Micro Cracks.* EU PVSEC. https://consensus.app/papers/details/w:bfa94065-a201-48aa-a0f0-3de9ec4c71fd/
- [R3] Kajari-Schröder S. et al. *Spatial and orientational distribution of cracks in crystalline photovoltaic modules generated by mechanical load tests.* https://www.researchgate.net/publication/236151699
- [R4] *Evaluating Solar Cell Fracture as a Function of Module Mechanical Loading Conditions* (LoadSpot, EL/I-V under load). https://www.researchgate.net/publication/318795700 ; *In-Situ Measurement of Power Loss for c-Si Modules Undergoing Thermal Cycling and Mechanical Loading.* https://www.researchgate.net/publication/347980280
- [R5] Paggi M. et al. (2014). *Fatigue degradation and electric recovery in Silicon solar cells embedded in photovoltaic modules.* Scientific Reports. https://consensus.app/papers/details/w:afe00a22-f8f0-406a-98f0-95e534ded3d1/
- [R6] Papargyri L. et al. (2020). *Modelling and experimental investigations of microcracks in crystalline silicon photovoltaics: A review.* Renewable Energy 145:2387–2408. https://consensus.app/papers/details/w:eb20ff5e-3bf2-48c1-94c2-96c0d837c41a/
- [R7] Ennemri A. et al. (2019). *Cracks in silicon photovoltaic modules: a review.* J. Optoelectron. Adv. Mater. 21(1-2):74–92. https://joam.inoe.ro/articles/cracks-in-silicon-photovoltaic-modules-a-review/
- [R8] Abdelhamid M., Singh R., Omar M. (2014). *Review of Microcrack Detection Techniques for Silicon Solar Cells.* IEEE J. Photovoltaics 4(1):514–524. https://ieeexplore.ieee.org/document/6648383
- [R9] Goudelis G., Lazaridis P., Dhimish M. (2022). *A Review of Models for Photovoltaic Crack and Hotspot Prediction.* Energies 15(12):4303. https://mdpi.com/1996-1073/15/12/4303

**Optical / imaging detection**
- [R10] Deitsch S. et al. (2019). *Automatic classification of defective photovoltaic module cells in electroluminescence images.* Solar Energy 185:455–468. https://arxiv.org/abs/1807.02894
- [R11] *Identifying defective solar cells in EL images using deep feature representations* (PeerJ CS 2022). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9138174/
- [R12] IEEE Access (2025), CNN from scratch on ELPV. https://doaj.org/article/dcd34e5b40e449dcbbccb20c87fb3e7a ; Buratti Y., *A Review of Deep Learning for Defects Classification … Luminescence Images.* https://apvi.org.au/solar-research-conference/wp-content/uploads/2022/02/Buratti-Yoann-A-Review-of-Deep-Learning-for-Defects-Classification-in-Silicon-Cells-and-Modules-from-Luminescence-Images.pdf
- [R13] Su B. et al. (2022). PVEL-AD dataset, IEEE Trans. Ind. Inform., DOI 10.1109/TII.2022.3162846. https://github.com/binyisu/PVEL-AD
- [R14] *Drone-Based Daylight Electroluminescence Imaging of PV Modules* (IEEE JPV 2020). https://orbit.dtu.dk/en/publications/drone-based-daylight-electroluminescence-imaging-of-pv-modules/
- [R15] Bhoopathy R. et al. (2018). *Outdoor photoluminescence imaging of photovoltaic modules with sunlight excitation.* Prog. Photovolt. 26:69–73; UAV PL: https://apvi.org.au/solar-research-conference/wp-content/uploads/2022/02/Scott-Alexander-Outdoor-Photoluminescence-Imaging-from-Unmanned-Aerial-Vehicles.pdf
- [R16] DTU (2026). Low-energy glass crack detection (daylight EL / motion-enhanced SWIR). https://orbit.dtu.dk/en/publications/a-novel-method-for-detecting-low-energy-front-glass-cracks-in-pho/ ; https://www.pv-magazine.com/2026/06/29/how-to-detect-low-energy-fractures-in-pv-modules/

**Active ultrasonic / piezo methods on PV**
- [R17] Silitonga D.J. et al. (2023). *Ultrasonic guided waves interaction with cracks in the front glass of thin-film solar PV module.* SOLMAT. https://www.sciencedirect.com/science/article/abs/pii/S0927024822005967
- [R18] Silitonga D.J. et al. (2024). *Front glass crack inspection of thin-film solar PV modules using high-order ultrasonic Lamb waves.* Solar Energy. https://consensus.app/papers/details/w:b413d5a6-c80b-4ec0-8783-f874831688f3/
- [R19] (2025). *A novel internal crack detection method for PV panels based on the air-coupled ultrasonic.* Sensors & Actuators A. https://www.sciencedirect.com/science/article/abs/pii/S0924424725004480
- [R23] Beroual S. et al. (2023). *Electromechanical Impedance Simulation-Based Evaluation of Cracks in PV Solar Cells.* Arab. J. Sci. Eng. https://consensus.app/papers/details/w:40a113e4-3f8d-4f83-adb1-d6116378b0d8/
- [R24] *Resonance Ultrasonic Vibrations for in-line crack detection in silicon wafers and solar cells.* https://www.researchgate.net/publication/224445466
- [R29] (2025). *Effect of Material Attenuation on Lamb Wave Propagation in a Solar Cell Array.* SHM 2025. https://dpi-proceedings.com/index.php/shm2025/article/view/37561

**Passive AE on cells / silicon / glass**
- [R20] Takabayashi Y. et al. (2022). *AE Monitoring of Damage in Flexible Solar Cells under Biaxial Loading.* JSME. https://consensus.app/papers/details/w:add2b21f-97d6-4ce8-a05f-a42b86098956/
- [R21] Tazawa H. et al. (2013). *Evaluation of Damage of Flexible Solar Cells under External Load using AE Method.* JSME Kanto. https://consensus.app/papers/details/w:a1f57c57-c7cf-4a64-9ca4-355d82ef5555/
- [R22] (2015). *Determination of Initial Crack Strength of Silicon Die Using Acoustic Emission Technique.* J. Electron. Mater. https://www.aminer.org/pub/56d8f060dabfae2eee716aba
- [R25] Thermal-cycling crack mechanisms (solder joints, ECA shingles). https://core.ac.uk/download/492500444.pdf ; https://publikationen.bibliothek.kit.edu/1000142274/149449686
- [R26] PVEL Mechanical Stress Sequence. https://www.kiwa.com/us/en-us/about-kiwa/entities/pvel/pqp/mss/
- [R27] Challenging Glass, AE for initial crack location in glass panes (via search summary; full citation to confirm).
- [R28] US Patent 8,023,110, *Priori crack detection in solar photovoltaic wafers…* https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8023110

**Localization**
- [R30] Kundu T. (2014). *Acoustic source localization.* Ultrasonics 54(1):25–38. https://experts.arizona.edu/en/publications/acoustic-source-localization/
- [R31] Park W.H., Packo P., Kundu T. (2016) and Nakatani H., Kundu T., Takeda N. (2014), anisotropic-plate cluster TDOA. https://experts.arizona.edu/en/publications/acoustic-source-localization-in-an-anisotropic-plate-without-know/ ; 2024 review: https://www.mdpi.com/2076-3417/14/19/8684
- [R32] Pearson M.R. et al. (2016). *Improved AE source location during fatigue and impact events in metallic and composite structures* (AIC-Delta-T). SHM. https://orca.cardiff.ac.uk/94482/
- [R33] Al-Jumaili S.K. et al. (2016). *AE source location in complex structures using full automatic Delta-T mapping.* MSSP. https://orca.cardiff.ac.uk/83785/1/1-s2.0-S0888327015005439-main.pdf
- [R34] Ebrahimkhanlou A., Salamone S. (2017). Single-sensor AE localization via multipath edge reflections. Ultrasonics.
- [R35] Ebrahimkhanlou A., Salamone S. (2018). *Single-Sensor AE Source Localization in Plate-Like Structures Using Deep Learning.* Aerospace 5(2):50. https://dx.doi.org/10.3390/aerospace5020050
- [R36] Ebrahimkhanlou A., Dubuc B., Salamone S. (2019). *A generalizable deep learning framework for localizing and characterizing AE sources in riveted metallic panels.* MSSP. https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/A-generalizable-deep-learning-framework-for/991021889906804721
- [R37] US Patent 6,933,930, *Accurate interactive acoustic plate* (interlayer damping of A-modes in laminated glass). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6933930
- [R38] *Theoretical and experimental studies of Lamb wave propagation in attenuative composites* (SPIE 6529). https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/6529/652929/

**ML for AE**
- [R39] Ciaburro G., Iannace G. (2022). *Machine-Learning-Based Methods for Acoustic Emission Testing: A Review.* Applied Sciences. https://consensus.app/papers/details/w:5966bad9-8220-40cd-860c-c706581d9964/
- [R40] Sikdar S. et al. (2021). *AE data based deep learning approach for classification and detection of damage-sources in a composite panel.* Composites Part B. https://consensus.app/papers/details/w:0258d096-234b-4b51-9262-6dc1e989a5b2/
- [R41] Li D. et al. (2020). *AE wave classification for rail crack monitoring based on synchrosqueezed wavelet transform and multi-branch CNN.* SHM. https://consensus.app/papers/details/w:22018141-bfc5-4eb6-9d15-3524c6b038a8/
- [R42] Unsupervised AE clustering (PCA + k-means, SOM) in glass-fibre laminates. https://pmc.ncbi.nlm.nih.gov/articles/PMC6439274 ; https://research.tudelft.nl/en/publications/multivariable-signal-processing-for-characterization-of-failure-m
- [R43] Krishna Y.S. et al. (2024). *Damage mode classification in CFRP laminates using convolutional autoencoder and CNN on AE waveforms.* SHM. https://consensus.app/papers/details/w:ab038cfe-86d4-4b63-819a-46e876bbebc6/
- [R44] (2024). AE noise vs. true AE separation (friction, rain, wind). Arab. J. Sci. Eng. https://link.springer.com/article/10.1007/s13369-024-09101-7 ; rain acoustic classification: https://arxiv.org/pdf/1804.05502
- [R45] (2023). *Source localization and transfer learning with unsupervised domain adaptation* (FE-generated AE). MSSP. https://www.peeref.com/works/27519850
- [R46] (2024). *Deep learning assisted SHM: AE analysis and domain adaptation.* Eng. Res. Express. https://impact.ornl.gov/en/publications/deep-learning-assisted-structural-health-monitoring-acoustic-emis/
- [R47] ASNT proceedings: FEM-pretrained AE localization with fine-tuning. https://source.asnt.org/1vofmq8/74 ; MSU dissertation (multi-fidelity GAN/Inception). https://d.lib.msu.edu/etd/52291
- [R48] (2026). *A Multi-Fidelity Convolutional Autoencoder-Transfer Learning Framework for Guided-Wave-Based Damage Diagnosis.* arXiv. https://arxiv.org/pdf/2606.27304
- [R49] Kita et al., InterNoise: sim-to-real gap for simulation-trained DNNs. https://author.ioa.org.uk/conference-proceeding/internoise-proceedings-v3/338-kita
- [R50] (2024). *Comparison of Tiny Machine Learning Techniques for Embedded Acoustic Emission Analysis.* arXiv 2411.17733. https://arxiv.org/html/2411.17733v1 ; tiny DL TOA localization (Computers 2023). https://cris.unibo.it/bitstream/11585/932536/1/computers-12-00129-v2.pdf
- [R51] (2023). Low-cost on-device PZT impact localization (PMMA). https://pubmed.ncbi.nlm.nih.gov/36679690/

**Hail dynamics**
- [R52] DuraMAT webinar (Dec 2023), hail impact dynamics (module modes ~11.5/14.4 Hz). https://www.duramat.org/docs/duramatlibraries/assets/webinars/duramat-webinar-december-2023.pdf ; Hartley et al., DIC of module deformation during hail impact.

**Standards referenced (not retrieved, cited from domain knowledge):** ASTM E976 (Hsu-Nielsen pencil-lead-break source), IEC 61215 (static mechanical load, hail test), IEC TS 62782 (dynamic mechanical load).
