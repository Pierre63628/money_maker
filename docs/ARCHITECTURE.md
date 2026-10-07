# Architecture et roadmap

**Priorité absolue :** intégrité des données → précision du backtest → robustesse → risque →
performance → automatisation.

## Flux cible

```
DISCOVER → BACKTEST → VALIDATE → STRESS TEST → RANK → PAPER TRADE → MONITOR → [APPROBATION MANUELLE] → LIVE
```

```
Market Data ─► Features ─► Strategy (poids cibles) ─► Risk Engine (bloque/réduit) ─► Portfolio ─► Broker
                                                                                               ├─ Backtest (simulé)
                                                                                               ├─ PaperBroker
                                                                                               └─ LiveBroker (jamais sans accord)
```

La stratégie ne voit jamais l'equity, les fills ni les limites. Remplacer `PaperBroker` par
`LiveBroker` ne touche pas la stratégie.

## Modules (`src/quant/`)

| Module | Rôle | État |
|---|---|---|
| `data/binance_vision.py` | Téléchargement Binance Vision : listing S3 paginé, checksum SHA256 obligatoire, CSV avec ou sans header, ms/µs | ✅ |
| `data/bybit.py` | Klines Bybit REST (audit des stratégies backtestées sur Bybit) | ✅ |
| `data/store.py` | Parquet par (exchange, market, dataset, interval, symbol), écriture atomique | ✅ |
| `data/validation.py` | Rapport qualité : doublons, conflits, grille, trous, OHLC, prix ≤ 0, volume nul, moves extrêmes | ✅ |
| `data/market_data.py` | Frames wide alignées sur une grille régulière, funding mappé sur les barres, masque d'univers, rééchantillonnage (barres complètes seulement) | ✅ |
| `data/binance_rest.py` | Réparation des trous via REST après recoupement avec Vision (`source = 1`), extension du funding récent, klines spot | ✅ |
| `data/binance_vision.py` (metrics) + `scripts/download_metrics.py` | Open interest et ratios de positionnement (5 min, archives quotidiennes), bornés par la période de cotation, nettoyage des valeurs nulles corrompues | ✅ |
| `data/universe.py` | Univers point-in-time sans survivorship : classification exchangeInfo (crypto / TradFi / index), éligibilité décalée d'un jour (âge, activité, liquidité médiane, top N) | ✅ |
| `features/core.py` | Indicateurs causaux (SMA, EMA, RMA Wilder, ATR, RSI, Donchian, z-score, vol) | ✅ |
| `signals/latch.py` | Machine à états entrée/sortie → position | ✅ |
| `strategies/base.py` | Interface `Strategy.target_weights(data)` + provenance | ✅ |
| `strategies/baselines.py` | Buy & hold, SMA cross, TSMOM, Donchian, RSI reversion | ✅ |
| `strategies/trend.py`, `cross_sectional.py`, `mean_reversion.py` | Trend avec hystérésis et vol-targeting ; momentum/réversal cross-sectional ; carry funding ; RSI(2) (source Strategy Factory) | ✅ |
| `backtesting/engine.py` + `_core.py` | Moteur vectoriel/numba (cf. `BACKTEST_CONVENTIONS.md`) | ✅ |
| `backtesting/event_engine.py` | Moteur événementiel : ordres stop/limite, chemin intrabarre défavorable, parité avec le paper | ⏳ P4 |
| `evaluation/metrics.py` | Toutes les métriques demandées + comparaison | ✅ |
| `evaluation/lookahead.py` | Détecteur générique de look-ahead par troncature | ✅ |
| `evaluation/fill_audit.py` | Audit des fills intrabarre de trade lists externes | ✅ |
| `evaluation/stats.py` | PSR, Deflated Sharpe, MinTRL, bootstrap stationnaire, PBO (CSCV) | ✅ |
| `evaluation/validation.py` | Walk-forward avec coûts de transition, perturbation de paramètres, régimes causaux, stress coûts | ✅ |
| `evaluation/registry.py` | Registre append-only de tous les essais (alimente le Deflated Sharpe) | ✅ |
| `research/` | Expériences pré-enregistrées, runner, portes de promotion, holdout unique | ✅ |
| `integrations/strategy_factory.py` | Client lecture seule (search, résumés, trade lists) | ✅ |
| `risk/engine.py` | Checks pré-trade avec métriques par ordre, blocage/réduction, limites DD (kill switch à reset manuel), perte journalière, poids, cluster corrélé, levier, liquidité, gap | ✅ (à brancher sur le moteur événementiel/paper) |
| `portfolio/sizing.py` | Rebalancement planifié, vol targeting, normalisation | ✅ partiel |
| `brokers/profiles.py` | Profils de courtier (instruments, spreads, swaps, commissions) chargés en JSON, pour revalider une stratégie sur un autre lieu d'exécution ; `brokers/mt5.py` documenté et désactivé | ✅ |
| `execution/orders.py`, `brokers/` | Ordres à id déterministe (idempotence), interface `Broker`, `PaperBroker` persistant (fills au bid/ask observé, journal, ledger), `LiveBroker` volontairement désactivé | ✅ paper / ⛔ live |
| `live/feed.py`, `live/loop.py` | Feed public (Vision + queue REST, funding), boucle paper : kill switch, idempotence par barre, steps en retard ignorés, réconciliation ledger/broker, risk engine, journal | ✅ |
| `models/` | ML (seulement s'il bat les baselines OOS après coûts) | ⏳ P6 |
| `monitoring/journal.py`, `scripts/paper_report.py` | Journal JSONL, alertes, rapport paper (slippage, latence, calibration du spread) | ✅ partiel (réplay backtest vs paper à faire) |

## Roadmap priorisée

| Phase | Contenu | Critère de sortie |
|---|---|---|
| **P0 Données** ✅ | 679 perps crypto (délistés inclus), classification, univers point-in-time, réparation REST vérifiée, masquage des barres figées post-délisting. **Reste :** calibration du spread sur carnet | Atteint |
| **P1 Backtester** ✅ | Exécution t+1, coûts (spread dynamique), funding, levier, liquidation, délisting et trous longs, cap liquidité, métriques, benchmarks, test look-ahead | Atteint (invariant comptable testé) |
| **P2 Validation** ✅ | Walk-forward, holdout gelé (2026-01 → 2026-09), Deflated Sharpe (registre), PBO, bootstrap stationnaire, sensibilité des paramètres, régimes, stress coûts, portes de promotion | Atteint |
| **P3 Recherche** 🔄 | 22 expériences, 129 essais, **aucune promotion**. Plus robuste : cash-and-carry taker (Sharpe OOS 4,6, CAGR 7,7 %, MDD −6 %, 5/6 portes, dormant depuis 2022). Meilleur directionnel : multi-sleeve maker (1,16). Rejetés : réversion, lead-lag, saisonnalité, cascades. **À faire :** données de positionnement (OI), stat-arb | Au moins une stratégie passe toutes les portes |
| **P4 Risque + exécution** 🔄 | Risk engine bloquant ✅ (branché sur le paper). Exécution maker simulée ✅ (traversée de la limite, haircut de queue, antisélection, fallback taker). **Reste :** stops intrabarre (moteur événementiel sur données 1m ou tick) | Parité < 1 bp/trade |
| **P5 Paper trading** 🔄 | Infrastructure ✅ et premier step live OK (shadow). **Reste :** exécution planifiée continue, réplay backtest vs paper | Plus de 4 semaines de paper stable, divergence expliquée |
| **P6 ML** | Features causales, purged CV, comparaison aux baselines | Gain OOS après coûts, sinon abandon |
| **P7 Live-ready** | Kill switch, limites de perte, réconciliation, reprise après crash, alertes. **Aucune connexion réelle sans accord explicite** | Revue manuelle |

## Registre des risques

| Risque | Mitigation |
|---|---|
| Look-ahead / leakage | Exécution t+1 imposée, détecteur par troncature sur toutes les stratégies, features causales testées |
| Survivorship | Univers construit depuis le listing Binance Vision (délistés inclus), sorties forcées au délisting |
| Data snooping / overfitting | Registre de tous les essais, Deflated Sharpe, PBO, OOS gelé, paramètres textbook d'abord |
| Coûts sous-estimés | Taker + spread + slippage + impact + funding, stress ×2, cap de liquidité |
| Artefacts intrabarre | Moteur vectoriel sans stops ; moteur événementiel avec chemin défavorable ; audit des fills |
| Données corrompues | Checksums, rapport qualité bloquant, aucune réparation silencieuse |
| Changement de régime | Résultats par année et par régime ; walk-forward |
| Risque opérationnel live | Paper d'abord, kill switch, idempotence, réconciliation, approbation manuelle |
