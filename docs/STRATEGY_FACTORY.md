# Strategy Factory (trader.dev) : analyse, audit et règles d'usage

Snapshot du 2026-09-14. La plateforme évolue : relancer `scripts/audit_strategy_factory.py`.

## Ce que c'est

- **Plateforme :** leaderboard public de stratégies Pine Script (TradingView), environ 240 800
  stratégies et 1 014 000 backtests.
- **Marchés :** majoritairement perps crypto Bybit/Binance, aussi XAUUSD et EURUSD.
- **Moteur :** émulateur « TradingView-like » (`engineVersion tv_jul26_mc7`). Commission 0,05 % par côté,
  **pas de slippage**, sizing 100 % de l'equity par défaut.
- **Accès sans compte :**
  - `GET https://mcp-api.strategyfactory.ai/strategies/search` : recherche filtrable ;
  - `GET https://mcp-api.trader.dev/backtest-results/{resultId}` : résumé détaillé ;
  - `r2Url` : trade list complète et courbe d'equity, en JSON gzip public.
- **Accès restreint :** le code source (`/backtest-results/{id}/source`, `fork.json`) exige la clé API du
  propriétaire ou un compte payant. **Non utilisé.** Aucune dépense n'a été engagée.

## Pourquoi le leaderboard n'est pas une preuve

1. **Multiple testing massif.** Avec 1M de backtests, le haut du classement est dominé par le biais de
   sélection, même sans aucun bug.
2. **Fenêtres courtes.** Beaucoup de backtests couvrent 3 à 30 jours (ex. LITUSDT : +1 842 % en 30 jours,
   Sharpe 25).
3. **Coûts sous-estimés :** pas de slippage, et funding absent des rapports.
4. **Artefacts de fills intrabarre** (voir l'audit ci-dessous).

## Audit 1 : sorties intrabarre (`quant.evaluation.fill_audit`)

- **Méthode :** pour chaque trade, on situe le prix de sortie dans le range de la barre de sortie, sur les
  OHLC Bybit du même marché : `1.0` = meilleur prix possible de la barre.
- **Référence :** une sortie sans connaissance du futur tombe dans les 5 % favorables environ 5 % du temps.
- **Verdict :** « suspect » si ce taux dépasse 35 % avec une p-value binomiale < 1e-6.

**Top Sharpe** (≥ 300 trades, DD ≤ 40 %, ≥ 180 jours) : **15/15 suspects.** 77 à 95 % des sorties se font
dans les 5 % favorables de la barre.

| Famille | Exemple | Affiché | Sorties au meilleur 5 % |
|---|---|---|---|
| SOL Donchian 15m + trailing ATR×0.03 | FTMO Guardrailed SOL Donchian | +4 700 %, PF 10, DD 0,5 % | 90–92 % |
| SOL 15m trailing 1 tick | Kontrolle B | +3 400 %, PF 12,7 | 77–78 % |
| AVAX Donchian 15m | MC Donchian transfer | +6 250 %, PF 9 | 88 % |
| BTC 2h EMA-VWAP + trail 0.005 ATR | Sweep | +19 700 %, PF 14,6, DD 2,2 % | 80 % |

**Explication.** Un trailing stop de quelques ticks, simulé sur des barres, suppose que le prix a
d'abord touché l'extrême favorable puis s'est retourné dans la même barre. En réel, le high n'est connu
qu'après coup. Ces rendements sont **des artefacts du simulateur, pas des edges.**

**Top profit** (≥ 200 trades, DD ≤ 60 %) : 7 suspects, 7 sans artefact intrabarre évident, 1 hors périmètre.

## Audit 2 : les « survivants » de l'audit 1

| Stratégie | Affiché | Entrées au close de la barre signal | Trades superposés | Profit net moyen/trade (0,05 %/côté, sans slippage) |
|---|---|---|---|---|
| Codex rebalanced LTC EMA9/VWAP 1h (5,9 ans), `01M2AC3CQEPH57EFSQKS68B1SC` | +19 970 % | 95 % | 25 % | 0,065 % |
| Vanta v7 RSI(2) mean-rev SUI 1h (2,1 ans), `01M2AC3J2MJ966Q8EC18EX9ZZD` | +19 740 % | 98 % | 26 % | 0,18 % |
| Codex Vanta Conservative (LTC 1h), `01M1GY2BDZR29HFJT3VR5YVM1N` | +19 430 % | 93 % | 35 % | −0,30 % |

**Constats :**
- **Fill au close de la barre signal.** Le signal est calculé sur le close et exécuté à ce même close,
  ce qui est irréaliste. Une variante s'appelle d'ailleurs « …process_orders_on_close ». Notre moteur
  exécute à l'open suivant.
- **P&L affiché non réconciliable** avec la trade list. Composer les `profitPct` donne +218 % au lieu de
  +19 970 % pour LTC, et −100 % au lieu de +19 430 % pour « Conservative ». Les positions superposées
  (pyramiding × 100 % equity) créent un levier caché.
- **Edge par trade du même ordre que les coûts réels :** frais 10 bps aller-retour + spread/slippage
  4–10 bps sur les altcoins + funding.

**Candidat retenu pour réimplémentation (P3) :** mean reversion RSI(2) 1h sur altcoins liquides
(source : `strategy_factory:01M2AC3J2MJ966Q8EC18EX9ZZD`, « Vanta v7 RSI(2) mean-rev SUI 1h »). Il sera exécuté à l'open suivant, avec coûts
réalistes, en walk-forward multi-actifs. Hypothèse : réversion courte sur les altcoins à forte
volatilité. **Risque principal :** l'edge disparaît après coûts.

## Règles d'usage

1. Strategy Factory est une **source d'idées**, jamais une preuve.
2. Toute réimplémentation porte `provenance = "strategy_factory:<resultId>"` et documente ce qui a été
   changé : exécution, sizing, coûts, filtres.
3. **Avant réimplémentation**, faire passer les audits 1 et 2 à la trade list publique.
4. Une stratégie ne vaut que par ses performances **dans notre moteur**, hors échantillon, après coûts.
