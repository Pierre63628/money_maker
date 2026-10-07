# Journal de recherche

Chaque entrée consigne ce qui a été testé, **combien d'essais**, sur quelles données, et ce qu'on en
conclut. Les essais comptent pour le Deflated Sharpe (P2) : ne rien supprimer.

---

## 2026-09-14 : baselines 1h, BTC/ETH/SOL perps (in-sample, non optimisé)

- **Données :** Binance USDT-M 1h, du 2020-01-01 au 2026-09-01 (58 440 barres), avec funding. SOLUSDT
  est listé depuis 2020-09-14.
- **Portefeuille :** allocation égale 1/3 par symbole, levier brut ≤ 1.
- **Coûts ×1 :** taker 5 bps, demi-spread 0,5 bp (BTC, ETH) et 1 bp (SOL), slippage 1 bp, impact 0,5.
- **Essais :** 9 stratégies × 3 niveaux de coûts (×0, ×1, ×2), plus 1 variante de période (voir point 4). Paramètres fixés a priori (valeurs
  classiques), **aucune optimisation**.
- **Commande :** `scripts/run_baselines.py --interval 1h`.

| Stratégie | Rendement ×0 | Rendement ×1 | CAGR ×1 | Sharpe ×1 | MDD ×1 | Trades | Turnover/an | Rendement ×2 |
|---|---|---|---|---|---|---|---|---|
| Buy & hold (perp, benchmark) | +2 542 % | +1 834 % | 56 % | 0,97 | −86 % | – | 0,4 | +1 829 % |
| SMA 24/168 long/short | +825 % | +180 % | 17 % | 0,55 | −66 % | 1 573 | 160 | +31 % |
| SMA 24/168 long only | +3 114 % | +1 164 % | 46 % | 1,04 | −59 % | 787 | 82 | +768 % |
| SMA 50j/200j long/short | +650 % | +258 % | 21 % | 0,62 | −76 % | 33 | 4 | +251 % |
| TSMOM 7j | +3 391 % | +42 % | 5 % | 0,38 | −67 % | 5 654 | 563 | −90 % |
| TSMOM 30j | +47 212 % | +6 920 % | 89 % | 1,37 | −47 % | 2 370 | 261 | +2 089 % |
| Donchian 55/20 barres | +25 % | −67 % | −15 % | −0,08 | −88 % | 2 130 | 206 | −88 % |
| Donchian 10j/5j | +1 254 % | +498 % | 31 % | 0,77 | −56 % | 470 | 48 | +380 % |
| RSI(14) 30/70 reversion | −89 % | −96 % | −37 % | −0,91 | −96 % | 2 440 | 234 | −99 % |

**Rendements par année civile (coûts ×1) :**

| Stratégie | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mois) |
|---|---|---|---|---|---|---|---|
| Buy & hold | +163 % | +879 % | −76 % | +184 % | +68 % | −20 % | −16 % |
| TSMOM 30j | +113 % | +659 % | +82 % | +150 % | −12 % | +21 % | −11 % |
| SMA 24/168 long only | +78 % | +351 % | −43 % | +159 % | +47 % | −21 % | −9 % |
| Donchian 10j/5j | +42 % | +90 % | −1 % | +97 % | +13 % | −29 % | +40 % |

**Lecture :**
1. **Coûts et funding dominent le court terme.**
   - TSMOM 7j : +3 391 % sans coûts, +42 % avec, −90 % à ×2.
   - Tout ce qui dépasse environ 200× de turnover par an est fragile.
   - Le funding coûte environ 12 %/an aux longs (taux moyen 0,011 %/8h).
2. **La mean reversion RSI 1h sur les majors perd même sans coûts** (−89 %). À cet horizon, pas de
   réversion exploitable sous cette forme.
3. **TSMOM 30j est la meilleure baseline**, mais :
   - l'essentiel vient de 2020–2023 ;
   - 2024–2026 est plat à négatif ;
   - sa sensibilité aux coûts est forte (turnover 261×/an, dû aux flips autour de zéro).
   - **Hypothèse à tester :** hystérésis, bande neutre ou rebalancement quotidien pour réduire le
     turnover.
4. **Sensibilité à la période et à l'univers (1 essai supplémentaire).** TSMOM 30j sur BTC+ETH seuls,
   du 2021-01-01 au 2026-09-01 : +312 %, Sharpe 0,74, MDD −47 %, soit la moitié du Sharpe de la version
   2020, 3 actifs. Un edge robuste ne devrait pas dépendre autant du point de départ.
5. **Rien ici n'est une preuve d'edge.** Tout est in-sample, 9 essais, 3 actifs corrélés, un seul
   régime dominant (bull 2020–21). À valider en walk-forward et sur l'univers élargi (P2/P3).

---

## 2026-09-14 : audit Strategy Factory

Voir `docs/STRATEGY_FACTORY.md`. Top Sharpe : 15/15 artefacts de fills intrabarre. Top profit : 7/14
suspects ; les 7 autres ont des entrées au close de la barre signal et un P&L non réconciliable.
Candidat retenu : RSI(2) 1h sur altcoins (à réimplémenter en P3).

---

## 2026-09-14 : P0 univers complet et intégrité

- **Téléchargement :** 679 perps USDT crypto en 1h + funding, **délistés inclus**. 185 perps non-crypto
  exclus via `exchangeInfo.underlyingType` : actions, ETF, métaux, indices, pré-IPO. Checksums vérifiés,
  0 échec.
- **Piège découvert :** Binance Vision continue de publier des **barres figées à volume nul après le
  délisting**, parfois pendant des années (FTMUSDT : 14 775 barres ; 158 symboles concernés). Sans
  traitement, un actif mort apparaît comme « listé » avec une volatilité nulle. Correctif : au chargement,
  les runs de 24 barres figées ou plus passent en NaN, ce qui déclenche la sortie forcée du moteur.
- **Trous :** 95 trous comblés via REST (5 688 barres), après recoupement exact avec Vision. 11 trous
  sont absents aussi de REST (vraies suspensions, ex. ICP septembre 2022, TLM mars 2023).
- **Univers point-in-time vérifié à la main :** en mai 2022, le top-10 contient LUNA, GMT et APE.
- **Benchmark clé :** le panier top-10 liquide acheté et conservé (sans survivorship) fait −57 %
  sur 2021–2025, alors que BTC fait +61 %. Suivre les alts « à la mode » détruit du capital.

## 2026-09-14 : batch 1 (5 expériences pré-enregistrées, 80 essais)

- **Protocole :** recherche du 2020-01-01 au 2026-01-01 ; walk-forward avec 365 jours d'entraînement et
  90 jours de test ; holdout 2026-01 → 2026-09 **non consommé**.
- **Coûts :** taker 5 bps, spread dynamique selon le volume, slippage 1 bp, impact, funding.

| Expérience | Candidats | Positifs in-sample | Sharpe OOS | DSR | PBO | Verdict |
|---|---|---|---|---|---|---|
| trend_top10_4h | 24 | 22 | 0,46 | 0,34 | 0,20 | REJECT |
| xsmom_top50_4h | 16 | 15 | 0,38 | 0,23 | 0,38 | REJECT |
| carry_top50_4h | 16 | 13 | 0,79 (MDD −80 %) | 0,52 | 0,55 | REJECT |
| rsi2_top30_1h (Strategy Factory) | 12 | **0** | −1,46 | 0,00 | 0,08 | REJECT |
| xsrev_top30_1h | 12 | **0** | −1,85 | 0,00 | 0,31 | REJECT |

**Diagnostic des coûts** (meilleur candidat in-sample, décomposition des frictions ; diagnostic, pas sélection) :

| Stratégie | Sharpe coût 0 | Frais seuls | Réaliste | Turnover/an |
|---|---|---|---|---|
| RSI(2) 1h | 2,86 | 0,97 | −0,61 | 1 100 |
| Réversal XS 1h | 1,42 | −0,07 | −1,30 | 1 100 |
| Momentum XS 4h | 1,24 | 1,12 | 1,18 | 92 |
| Trend 4h | 1,20 | 1,18 | 1,06 | 31 |

**Conclusions :**
1. **La réversion court terme existe en brut** (Sharpe 2,9 à coût nul), mais elle est inférieure au
   coût taker. Elle n'est capturable qu'avec une exécution maker (ordres limites, frais maker), à
   étudier avec des données tick et une modélisation de l'antisélection. **Le RSI(2) de Strategy Factory
   est falsifié en exécution taker réaliste.**
2. **Les signaux lents survivent aux coûts**, mais la sélection de paramètres en walk-forward est
   instable (stabilité 0,25–0,35) et dégrade le hors échantillon.
3. **Le carry funding collecte bien le funding**, mais subit des squeezes sur les shorts (MDD −80 à −90 %).

## 2026-09-14 : batch 2 (enregistré APRÈS le batch 1, 4 essais de plus, N total = 84)

Hypothèse : supprimer la sélection (ensemble de toute la grille), puis diversifier entre familles.

| Expérience | Sharpe OOS | CAGR | MDD | DSR | Verdict |
|---|---|---|---|---|---|
| trend_ensemble_top10_4h | 0,53 | – | – | 0,10 | REJECT |
| xsmom_ensemble_top50_4h | 0,69 | – | – | 0,18 | REJECT |
| carry_ensemble_top50_4h | −0,06 | – | – | 0,00 | REJECT |
| **multisleeve_top30_4h** (risk parity trend+xsmom+carry) | **1,05** | 32 % | −50 % | 0,45 | REJECT (4/6 portes) |

**Multi-sleeve :**
- **Résultats :** +304 % OOS 2021–2025 ; bootstrap P(Sharpe ≤ 0) = 0,01 ; Sharpe 0,74 à coûts ×2.
- **Échecs :** Deflated Sharpe 0,45 et MDD −50 %.
- **Déclin :** +209 % (2021), −4 % (2022), +80 % (2023), **−18 % (2024), −9 % (2025)**. L'edge semble
  décliner.
- **Statut :** **non promu.** Holdout non consommé.

**État honnête : aucune stratégie ne démontre un edge robuste après coûts à ce stade.**

## 2026-09-14 : infrastructure paper trading (P5), premier step live en shadow

- **Stratégie :** `multisleeve_top30_4h` exécutée en **shadow**. Elle n'est pas promue : il s'agit de
  valider l'infrastructure, pas de prouver un edge.
- **Chaîne :** feed public Binance (sans compte) → univers point-in-time → stratégie (poids identiques au
  calcul de recherche, test de parité) → risk engine → PaperBroker (fill au bid/ask observé) →
  journal et réconciliation.
- **Step du 2026-09-14 (barre 12:00 UTC) :** 30 symboles éligibles, 12 fills, notionnel 3 222 $,
  coût d'exécution moyen 2,2 bps.
- **Calibration préliminaire :** demi-spread observé / modélisé = **0,13** en médiane (p90 0,57). Le
  modèle de spread du backtest est conservateur au top-of-book pour de petites tailles. Échantillon trop
  petit pour modifier le modèle, et la profondeur n'est pas mesurée.
- **Latence :** step lancé à la main 18 min après la clôture de la barre. Le glissement contre le close
  de décision atteint 47 bps en moyenne signée. En production, le step doit tourner dans la minute qui
  suit la clôture (mode `--loop`).

## 2026-09-14 : batch 3, exécution maker (enregistré après la décomposition des coûts)

- **Modèle d'exécution :** ordre limite au close de décision ±1 bp. Fill uniquement si la barre suivante
  traverse la limite de 2 bps (haircut de queue). Frais maker 2 bps. Ordre marketable au gap → taker.
  Sorties non remplies → taker après 3 barres. Stress ×2/×3 : frais maker et haircut multipliés aussi.

| Expérience | Taker (batch 1–2) | Maker | Verdict |
|---|---|---|---|
| rsi2_top30_1h | −1,46 | −0,26 | REJECT |
| xsrev_top30_1h | −1,85 | −1,09 | REJECT |
| multisleeve_top30_4h | 1,05 | **1,16** (DSR 0,55) | REJECT |

**Décomposition RSI(2)** (meilleur candidat in-sample, bps par trade) : brut 18,7 → maker idéal 18,1 →
avec haircut de queue et fills manqués **12,3** (antisélection ≈ 6 bps) → avec frais maker et fallbacks
**5,3**. Avec des frais maker nuls (VIP ou market maker), le Sharpe in-sample serait de 0,95. **La
réversion 1h n'est pas exploitable aux frais retail. Piste abandonnée.**

## 2026-09-14 : batch 4 (paramètres a priori)

| Expérience | Sharpe OOS | DSR | PBO | Sharpe brut (coût 0, meilleur IS) | Verdict |
|---|---|---|---|---|---|
| carry_filtered_maker_top50_4h | 0,59 | 0,67 | 0,71 | 0,64 | REJECT |
| multisleeve_v2_maker_top30_4h (frein DD) | 0,86 | 0,30 | n/a | 0,96 (MDD −29 %) | REJECT |
| seasonality_top5_maker_1h | −2,82 | 0,00 | 0,68 | 0,50 | REJECT |
| leadlag_top30_maker_1h | −3,58 | 0,00 | 0,77 | **4,19** (turnover 3 465×/an) | REJECT |

**Lecture :**
- Tous les signaux horaires testés (réversion, réversal cross-sectional, lead-lag, saisonnalité) ont
  un edge **brut** positif, parfois spectaculaire (lead-lag 4,2), qui disparaît dès les frais. Une
  partie de cet edge brut vient probablement du bid-ask bounce (signal au dernier prix, exécution à
  l'open suivant), non capturable.
- **Leçon de méthode :** tout Sharpe brut élevé à l'échelle horaire doit être présumé non capturable
  tant qu'il ne survit pas à l'exécution simulée.
- Le frein de drawdown réduit le MDD (−50 % → environ −30 %), au prix du rendement.

## 2026-09-14 : batches 5–7, événements de liquidation et combinaisons (N total = 125)

| Expérience | Sharpe OOS | MDD OOS | DSR | Verdict |
|---|---|---|---|---|
| cascade_maker_top50_1h (12 cand.) | −0,41 | −76 % | 0,00 | REJECT |
| idio_cascade_maker_top50_1h (8 cand., post-hoc) | −0,28 | −28 % | 0,00 | REJECT |
| cascade_ensemble_maker_top50_1h | 0,55 | −39,5 % | 0,07 | REJECT |
| multisleeve_v3_maker_top30_1h (+ cascade) | 0,91 | −51 % | 0,28 | REJECT |
| *rappel : multisleeve_maker_top30_4h* | *1,16* | *−47,5 %* | *0,55* | *REJECT (meilleur à ce jour)* |

**Réversion après cascade de liquidations** (move ≤ −kσ, volume ≥ 3× la médiane, ventes taker
dominantes) :
- **Edge par trade :** c'est la première famille dont l'edge par trade dépasse largement les coûts. 5σ,
  hold 24h, long : +218 bps/trade après exécution maker ; 11/12 candidats positifs in-sample.
- **Profil de risque :** toutes les variantes perdent en 2022 et 2025. Les pires trades sont les krachs
  systémiques (2022-05-11 LUNA : PEOPLE −48 %, DYDX −36 % ; 2025-10-10 : FARTCOIN −35 %) et des
  effondrements de tokens (BANANAS31 −56 %).
- **Hypothèse idiosyncratique falsifiée :** restreindre aux chocs idiosyncratiques (résiduel vs BTC,
  largeur de marché calme) détruit l'edge (Sharpe IS max 0,98 → 0,50).
- **Interprétation :** la prime vient des rebonds **après les krachs de marché**. Économiquement, c'est
  une vente d'assurance crash (short convexité) : un gain régulier et des pertes rares mais massives.
  C'est une prime de risque, pas un alpha gratuit.
- **Combinaison avec le trend (long convexité) :** n'améliore pas le multi-sleeve (0,91 contre 1,16).

## Synthèse au 2026-09-14 (18 expériences, 125 essais)

1. **Aucune stratégie ne franchit les portes.** Le holdout 2026-01 → 2026-09 n'est pas consommé.
2. **Hiérarchie des edges observés** (hors échantillon, après coûts) :
   - facteurs lents combinés (trend + momentum XS + carry, exécution maker) : Sharpe environ 1,1,
     MDD −50 %, **négatifs en 2024–2025** ;
   - trend seul ou momentum XS seul : 0,4–0,7 ;
   - prime de liquidation : environ 0,5 en ensemble, avec queue gauche ;
   - tout signal horaire : edge brut réel mais inférieur aux coûts retail, même en maker.
3. **Déclin récent :** la plupart des edges se dégradent depuis 2024. Il faut soit des edges qui
   fonctionnent **sur 2024–2025**, soit accepter que les primes restantes sont modestes.
4. **Pistes restantes à justification économique :**
   - cash-and-carry spot/perp (delta-neutre, edge structurel, rendement lié au niveau du funding) ;
   - données de positionnement (open interest, ratios long/short : `metrics` Binance Vision) pour
     conditionner la prime de liquidation et le carry ;
   - calibration du coût réel via paper trading (le spread observé vaut environ 0,14 × le modèle).

## 2026-09-14 : batch 8, cash-and-carry spot/perp (N total = 129)

- **Données :** klines spot 1h pour 424 symboles du top-50 perp (337 ont une paire spot).
- **Chargeur de paires :** colonnes `SYMBOL` (perp) et `SYMBOL@spot`. Le funding s'applique à la seule
  jambe perp. La jambe spot doit avoir sa propre liquidité récente.
- **Stratégie :** long spot + short perp quand le funding annualisé sur 3 ou 7 jours dépasse 10 % ou
  20 %, sortie sous un tiers du seuil. 4 candidats a priori. Frais spot 10 bps, levier brut ≤ 1,6.

**Falsification, deux artefacts trouvés puis corrigés :**
1. **Jambe spot sans données récentes considérée « tradable »** (ffill illimité), d'où une jambe perp
   **nue** : REEF 2024, +363 $ directionnels. Correction : `MarketData.tradable` exige un prix dans les
   24 dernières barres (appliqué à toutes les expériences) ; la jambe spot a son propre filtre de
   liquidité.
2. **Délisting asynchrone des deux jambes :** BNX 2025, gain de basis fictif de +211 $.

Autre constat : exécuter chaque jambe en maker indépendamment crée du legging. Une paire exige une
exécution **atomique**, d'où la variante taker (enregistrée comme nouvel essai).

| Expérience (après correctifs) | Sharpe OOS | CAGR | Vol | MDD | PBO | DSR porte | DSR poolé (info) | Verdict |
|---|---|---|---|---|---|---|---|---|
| cash_carry_top50_4h (maker) | 3,82 | 7,5 % | 1,9 % | −7,1 % | 0,15 | 0,04 | – | REJECT |
| cash_carry_taker_top50_4h | **4,61** | 7,7 % | 1,6 % | −6,4 % | 0,00 | **0,00** | **0,96** | REJECT (5/6 portes) |

- **Décomposition** (meilleur candidat IS, 2020–2025) : funding +9 114 $, frais −1 283 $, basis −826 $.
  181 paires, top-5 = 17 % du PnL, pire jour −0,94 %.
- **Rendement concentré en 2021** (+33 %). Ensuite : 2022 +0,1 %, 2023 +1,9 %, 2024 +5,8 %,
  2025 +0,9 %. Le funding s'est comprimé, la stratégie est souvent inactive.
- **Dry-run live (2026-09-14) :** 38 paires tradables, **aucune position**. Funding médian sur 7 jours
  à 3,7 %/an ; les funding élevés sont sur des micro-caps hors univers.
- **Lecture :** edge **statistiquement réel** (Sharpe très élevé, drawdown faible, robuste aux coûts
  ×2) mais **économiquement modeste et dépendant du régime**. C'est une poche basse-risque, pas un
  moteur de rendement.

### Note méthodologique : Deflated Sharpe

- **Défaut de la spécification pré-engagée :** N = tous les essais, mais variance des Sharpe prise sur
  les seuls candidats de l'expérience. Pour une famille à très faible volatilité, un candidat faible
  gonfle la variance : seuil ≈ Sharpe annuel 5,8 pour le carry.
- **Variance poolée sur tous les essais :** trop stricte. Les essais détruits par les coûts
  (Sharpe −5) violent l'hypothèse nulle, ce qui donne un seuil ≈ Sharpe annuel 4 pour tout le monde.
- **Décision :** la porte garde la spécification **pré-engagée** (on ne change pas une règle après
  avoir vu un résultat favorable). Les deux variantes sont affichées à titre d'information.
- **Pour les expériences futures :** définir la spécification avant de les lancer, variance par
  famille de stratégies.

### Autres constats

- **Sélection instable :** un micro-changement (Sharpe in-sample −4,854 → −4,846) fait basculer la
  sélection walk-forward de xsrev_maker (OOS −1,09 → −2,85). Une sélection entre candidats tous
  mauvais est instable.
- **Bug live trouvé par dry-run et corrigé :** un symbole sans historique ni données REST faisait
  planter le feed. La boucle carry aurait échoué à chaque step.

## 2026-09-17 : batch 9, open interest et positionnement (N total = 153)

- **Données :** dataset `metrics` de Binance Vision, archives **quotidiennes uniquement**, pas de vue
  mensuelle. Open interest, ratio long/short des comptes retail, ratios des top traders, ratio taker.
  283 symboles, 2022→2026, couverture médiane 99,9 %.
- **Incident disque :** stockées en 5 minutes avec le cache brut, ces données saturaient le disque de
  la machine (qui est par ailleurs déjà rempli à 93 % indépendamment du projet). Correctif : stockage
  **rééchantillonné en 1h, float32, zstd**, purge du cache brut. **1,25 Go → 73 Mo** pour 80 symboles.
- **Nettoyage :** Binance publie parfois un open interest à zéro. Non masqué, cela crée des variations
  infinies, donc des signaux fabriqués.

| Expérience | Candidats | Sharpe OOS | MDD | DSR | PBO | Verdict |
|---|---|---|---|---|---|---|
| cascade_oi_top30_1h | 8 | 0,94 (contre −0,41 sans OI) | −37 % | 0,75 | **0,81** | REJECT |
| oi_trend_top30_4h | 8 | 0,07 | – | 0,00 | 0,37 | REJECT |
| **crowding_top30_4h** | 8 | **1,19** | −34,7 % | 0,14 | **0,03** | REJECT (5/6 portes) |

**Cascade + open interest :** exiger une baisse d'open interest (donc des liquidations forcées, pas de
nouveaux shorts) fait passer le Sharpe hors échantillon de −0,41 à 0,94. Mais PBO 0,81 : le meilleur
candidat in-sample ne généralise pas, la sélection est instable. Piste réelle, mise en attente.

**Contrarian de positionnement retail (`crowding_top30_4h`) — meilleur résultat du projet à ce jour :**

- **Signal :** z-score sur 90 jours du ratio long/short des **comptes retail** ; short les perps où le
  retail est anormalement long, long ceux où il est anormalement short ; rebalancement quotidien.
- **Hors échantillon 2023–2025 :** +178 %, CAGR 40,6 %, Sharpe 1,19, MDD −34,7 %, marché-neutre
  (corrélation BTC 0,03, exposition nette 1,7 %).
- **Régularité inhabituelle :** +55,7 % (2023), +14,8 % (2024), +55,5 % (2025). Aucune décroissance,
  contrairement à toutes les autres familles testées.
- **Portes :** 5 sur 6. Seul échec, le Deflated Sharpe (0,14), pénalisé par N = 153 essais cumulés.

**Falsification (tous les tests passés) :**
| Test | Résultat |
|---|---|
| Décalage des métriques de 4h / 8h / 24h | Sharpe 1,24 → 0,94 / 0,97 / 1,11 : pas un artefact d'horodatage |
| Même mécanique sur un z-score de **prix** | **−1,30** : ce n'est pas une réversion de prix déguisée |
| Ratios des **top traders** au lieu du retail | −0,04 à −0,57 : seul le « dumb money » porte le signal |
| Sensibilité des paramètres (20 voisins) | **20/20 positifs**, Sharpe médian 1,22, pire 0,91 |
| Capacité | Sharpe 1,24 à 10 k$, 1,16 à 1 M$, 1,00 à 10 M$, 0,90 à 50 M$ |
| Régimes | Positif partout : bear 0,80, bull 1,52, vol haute 0,87, vol basse 1,43 |
| Coûts ×2 / ×3 | Sharpe 0,75 / 0,35 (turnover 256×/an : sensible) |

**Réserve :** corrélation +0,52 avec le momentum cross-sectional, donc recouvrement partiel avec une
prime déjà connue.

**Confirmations en cours, sans toucher au holdout :** application de la configuration **figée** à
2020-09 → 2022 (période antérieure jamais utilisée) et à un univers plus large (top-50) ainsi qu'en
barres 1h (top-20).

## 2026-09-17 : batch 10, confirmations du signal de positionnement (N total = 155)

Contrainte de données : Binance ne publie les `metrics` qu'à partir de **2021-12** pour la quasi-totalité
des symboles (BTC depuis 2020-09). Un test sur une période antérieure est donc impossible ; les tests
indépendants disponibles sont un autre univers, une autre fréquence, le holdout et le paper forward.

Configuration **figée** (z-score 90 jours du ratio de comptes retail, rebalancement quotidien,
quintile 20 %), un seul candidat par expérience, donc **aucune sélection**.

| Univers / barres | Rendement OOS | CAGR | Sharpe | MDD | Coûts ×2 | Coûts ×3 | 2023 / 2024 / 2025 |
|---|---|---|---|---|---|---|---|
| top-30 4h (origine) | +178 % | 41 % | 1,19 | −35 % | 0,75 | 0,35 | +56 / +15 / +56 % |
| top-50 4h | +223 % | 48 % | **1,58** | **−21 %** | 1,18 | 0,74 | +71 / +9 / +73 % |
| top-20 1h | +331 % | 63 % | 1,53 | −22 % | 1,20 | 0,81 | +97 / +50 / +46 % |

- Bootstrap P(Sharpe ≤ 0) = 0,00 sur les deux confirmations.
- Les trois échouent **uniquement** la porte Deflated Sharpe (0,51 à 0,57 ; seuil 0,9), pénalisée par
  les 153 essais cumulés du programme. Une réplication à paramètres figés n'ajoute pourtant pas de
  risque de sélection : la porte est conservée telle quelle, mais cette limite est explicitée ici.
- **Statut :** non promue. Holdout 2026-01 → 2026-09 **toujours intact**. Paper forward en shadow lancé
  sur la variante top-50 4h.

### Chaîne live pour ce signal
- Endpoints publics `openInterestHist`, `globalLongShortAccountRatio`, `topLongShortAccountRatio`,
  `topLongShortPositionRatio`, `takerlongshortRatio` (environ 30 jours d'historique), complétés par le
  store Vision. Queue demandée **à la période de la barre**, pas en 5 minutes, sinon le nombre de
  requêtes fait exploser les timeouts.
- Bugs trouvés et corrigés en dry-run : doublons d'horodatage aux frontières de tranches REST,
  absence de reprise sur timeout, et plantage de la boucle quand la stratégie n'a aucune position.
- Dry-run live du 2026-09-17 : 70 symboles, couverture 70/70, 18 positions (9 longues, 9 courtes),
  exposition brute 1,0 et nette 0,0, chargement en 109 s.

## 2026-09-17 : HOLDOUT CONSOMMÉ sur le signal de positionnement (2026-01-01 → 2026-09-01)

Test unique, décidé par l'utilisateur. Variante **principale désignée avant lecture** :
`crowding_top50_4h`. Les deux autres variantes portent sur la même fenêtre : elles sont **secondaires
et non indépendantes**.

| Variante | Rendement (8 mois) | CAGR | Sharpe | MDD | Turnover |
|---|---|---|---|---|---|
| **top-50 4h (principale)** | **+38,1 %** | 62 % | 1,37 | −30,5 % | 210 |
| top-30 4h (secondaire) | +34,8 % | 57 % | 1,17 | −27,4 % | 243 |
| top-20 1h (secondaire) | +196,3 % | 411 % | 2,84 | −22,5 % | 239 |

Contexte : **BTC fait −10,5 % sur la même fenêtre**. Bêta mesuré −0,09, exposition nette 1,4 % :
c'est bien de l'alpha, pas une exposition directionnelle cachée.

**Mise en perspective (bootstrap de fenêtres de 8 mois tirées de la période de recherche) :**
- rendement : médiane +27,6 %, p5 −8,3 %, p95 +91,6 %, probabilité de perte 11 %.
  **Le +38 % observé est au 64ᵉ percentile : conforme.**
- drawdown : médiane −12,8 %, pire 5 % à −23,1 %. **Le −30,5 % observé est pire que 95 % des
  trajectoires.**

**Réserve majeure : concentration.**

| Fenêtre | Base | Sans les 3 meilleurs symboles | Part des 3 premiers | Symboles tradés |
|---|---|---|---|---|
| Recherche 2023-2025 | +223 % (Sharpe 1,57) | +137 % (1,28) | 18 % | 266 |
| Holdout 2026 | +38 % (1,36) | **−10,6 % (−0,34)** | 31 % | 129 |

Les trois gagnants du holdout sont des effondrements : LAB −39 %, SIREN −57 %, RIVER −84 %, tous
shortés pendant que les comptes retail s'y entassaient. **L'edge est à queue épaisse :** il se paie sur
de rares effondrements de tokens sur-détenus par le retail. Le trade médian est légèrement négatif.
Sur 8 mois, tout peut donc dépendre de la présence ou non de tels événements.

**Statut : toujours non promue** (la porte Deflated Sharpe reste échouée), mais le holdout **confirme**
l'edge et en précise le profil de risque : rendement élevé, neutralité au marché, drawdowns d'environ
30 %, gains concentrés, longues périodes plates possibles.

**Le holdout est consommé.** Toute amélioration (cap de poids par symbole, exclusion des cotations
récentes, frein de drawdown, combinaison avec les autres poches) devra être pré-enregistrée, testée sur
la seule période de recherche, puis validée en **paper forward**, seule source d'évidence restante.

## 2026-09-17 : batch 11, contrôles de concentration (N total = 157)

Enregistré **après** le holdout, donc évalué sur la seule période de recherche 2022-2026. Ajouts a
priori : pondération inverse à la volatilité dans chaque jambe, plafond par symbole, âge minimum de
cotation, rééquilibrage des jambes après plafonnement.

| Expérience | Sharpe OOS | CAGR | MDD | DSR | PBO | Verdict |
|---|---|---|---|---|---|---|
| **crowding_robust_top50_4h** | 1,22 | 32 % | **−19,6 %** | **0,835** | 0,34 | REJECT (5/6) |
| multisleeve_v4_maker_top50_4h (trend + xsmom + crowding) | 0,90 | – | – | 0,13 | n/a | REJECT |

- 4/4 candidats positifs in-sample ; le meilleur (plafond 5 %, sans filtre d'âge) fait Sharpe 1,37 et
  MDD −19,6 %.
- Le **Deflated Sharpe atteint 0,835**, meilleur score du projet, contre 0,14 pour la version non
  contrainte : réduire la concentration réduit aussi la pénalité statistique. Le seuil de 0,9 reste
  manqué de peu.
- Diversifier avec trend et momentum **dégrade** le résultat (0,90) : le signal de positionnement porte
  seul l'essentiel du rendement.

## 2026-09-17 : faisabilité MetaTrader 5 (demande utilisateur)

MT5 signifie des **CFD crypto chez un courtier**, pas les perpétuels Binance. Trois différences
décident du sort de la stratégie : les instruments listés, les spreads, et le **swap overnight, payé
sur les deux jambes** d'un portefeuille neutre (le funding des perpétuels, lui, est un transfert).

Ajouts au moteur : `financing_bps_per_day` par symbole (coût sur le brut), profils de courtier
(`BrokerProfile`, JSON issu des spécifications contractuelles), adaptateur MT5 volontairement
désactivé, et `scripts/broker_feasibility.py` qui calcule le **spread d'équilibre**.

| Univers | Binance maker | CFD 10 bps | CFD 10 bps + swap 4 bps/j | CFD 25 bps |
|---|---|---|---|---|
| top-50 | Sharpe 1,38 · CAGR 40 % | 0,64 · 14 % | 0,64 · 1 % | −0,64 · −19 % |
| top-20 | 1,77 · 30 % | 1,07 · 16 % | 1,07 · 10,5 % | 0,13 · 1 % |
| top-10 | 1,37 · 15 % | 0,84 · 9 % | 0,84 · 6 % | 0,18 · 1 % |

Avec un profil CFD générique à 10 instruments : **spread d'équilibre 13,5 bps par côté** pour le signal
de positionnement, 45 bps pour le trend. Conclusion : la faisabilité MT5 dépend surtout du **nombre de
paires crypto listées** (l'edge est cross-sectional) et accessoirement des spreads. Verdict définitif
impossible sans les spécifications réelles du courtier.

## 2026-10-06 : revue complète du pipeline (code, documentation Binance, mesures sur l'API)

Aucune erreur dans le moteur de backtest, les statistiques ou le protocole. **Quatre écarts entre ce que
la recherche a validé et ce que la boucle live exécutait**, tous corrigés :

| # | Écart | Effet | Correctif |
|---|---|---|---|
| 1 | Le live lisait le positionnement **une barre trop vieux** (tampons REST = fin de période, traités comme des débuts) | Sharpe attendu 1,16 → **0,86** | `snapshots_to_bars`, regroupement `(T, T+Δ]` ; test d'égalité recherche/live |
| 2 | Le live réalignait la dérive à **chaque barre** ; le backtest ne trade un symbole que si sa cible change | Écart de **parité**, pas de performance (voir ci-dessous) | Règle `on_change` par symbole, avec rattrapage d'une décision manquée ; l'ancien comportement reste disponible (`--rebalance drift`) |
| 3 | Funding d'un broker réel **compté plusieurs fois** puis retranché d'une équité qui l'incluait déjà | Tailles légèrement fausses, journal faux | Curseur persistant, équité relue auprès de la venue |
| 4 | Écritures de cache **non atomiques**, plusieurs boucles se réveillant à la même seconde | Fichier parquet corrompu (déjà observé une fois) | Écriture temporaire + renommage, cache illisible reconstruit |

Sensibilité du résultat à la fraîcheur du ratio (période de recherche, top-50 4h) :

| Instantané utilisé pour la barre T | Base maker | Base taker | Robuste maker | Robuste taker |
|---|---|---|---|---|
| Clôture `T+4h` (recherche) | 1,41 | 1,16 | 1,37 | 1,11 |
| `T+3h` | 1,37 | 1,06 | 1,33 | 1,06 |
| `T` (live avant correctif) | 1,17 | **0,86** | 1,15 | 0,83 |
| `T−4h` | 1,03 | 0,73 | 1,00 | 0,69 |

**Sur l'écart n°2, j'avais d'abord tort.** Je l'ai présenté comme un surcoût ; mesuré, le maintien des
poids n'est pas pénalisant. Enregistré comme expérience (`crowding_drift_top50_4h`, origine post-hoc
explicitée) et passé par le protocole :

| Fenêtre | `on_change` (validé) | Maintien des poids |
|---|---|---|
| 2022 (apprentissage du walk-forward) | Sharpe 0,84 | 1,60 |
| **2023-2025 (hors échantillon)** | **1,58**, CAGR 47,9 %, MDD −20,9 % | **1,64**, CAGR 45,7 %, MDD −19,5 % |

Hors échantillon les deux sont équivalents ; tout l'écart apparent (1,63 contre 1,41 sur la période
entière, répété sur 8 configurations) venait de 2022. **Aucune raison de changer** : le live est aligné
sur la configuration validée. Bon rappel de la règle : un diagnostic cohérent sur huit variantes de la
même période n'est qu'une seule observation.

Attentes pour le forward, sur la fenêtre hors échantillon 2023-2025 :

| Configuration | Sharpe | CAGR | MDD |
|---|---|---|---|
| Recherche validée (maker, `on_change`) | 1,58 | 48 % | −21 % |
| **Live corrigé (taker, `on_change`)** | **1,34** | **39 %** | −22 % |
| Live avant correctif (taker, données vieilles d'une barre) | 0,92 | 22 % | −24 % |

**Points vérifiés sans défaut :** signe et instant du funding, décalage d'exécution d'une barre,
volumes et volatilités retardés, univers point-in-time, formules PSR/DSR/PBO (Bailey et López de Prado),
walk-forward sans fuite, correspondance des colonnes de l'archive (`count_long_short_ratio` = comptes de
tous les traders).

**Ce que la boucle exécute réellement :** des ordres **au marché**. La référence pour juger le forward
est donc le backtest **taker** (Sharpe 1,16, CAGR 32 %), pas le backtest maker (1,41, 40 %). L'écart
est le gain attendu d'une exécution passive, non encore implémentée.

**Rafraîchissement hebdomadaire :** `--skip-existing` laissait les barres et le funding figés à la date
du premier téléchargement (seul le positionnement était prolongé). Remplacé par `--extend`.

**Écarts restants, assumés :**
- le moteur de risque plafonne certaines positions courtes très volatiles (cap de gap), ce qui laisse
  un biais net long d'environ 5 % absent du backtest ;
- le kill switch à −30 % est du même ordre que le drawdown historique (−21 à −35 %) : il se
  déclenchera probablement un jour en fonctionnement normal ;
- la variante déployée est la version de base (poids égaux) ; la variante robuste n'est pas en forward.

Les journaux antérieurs au correctif sont archivés : ils décrivent un autre comportement.
