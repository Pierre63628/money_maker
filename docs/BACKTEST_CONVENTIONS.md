# Conventions du backtester

Ce document est le contrat du moteur `quant.backtesting.engine`. Toute stratégie, tout test et tout
rapport doivent le respecter. Si une convention change, ce document et les tests changent avec elle.

## Temps et exécution

| Élément | Convention |
|---|---|
| Horodatage d'une barre | `t` = heure d'**ouverture**, UTC. La barre couvre `[t, t + intervalle)`. |
| Information disponible | Le close de la barre `t` n'est connu qu'à `t + intervalle`. |
| Décision | `weights.loc[t]` utilise uniquement les barres `<= t`. |
| Exécution | À l'**open** de la barre `t + execution_delay`. `execution_delay >= 1` est imposé (sinon `ValueError`). |
| Marquage | L'equity est marquée au close de chaque barre. Dans les métriques, elle est ré-horodatée à `t + intervalle`. |
| Barre manquante | Aucun trade possible. L'ordre reste en attente et s'exécute à la prochaine barre valide. Les positions sont marquées au dernier prix connu. |
| Délisting / trou long | Si les données d'un symbole s'arrêtent, ou si un trou de plus de `max_gap_bars` (72 par défaut) commence, avec une position ouverte : sortie forcée au dernier close avec une décote (`delist_penalty_bps`, 100 bps). Cela couvre aussi les tickers délistés puis réutilisés. |

## Rebalancement

- `on_change` (défaut) : un symbole n'est tradé que si son poids cible change, ou si le levier dérive
  au-delà de `max_gross_leverage × (1 + leverage_tolerance)`. Dans ce cas, tout est ramené aux cibles.
  Buy & hold reste donc buy & hold.
- `every_bar` : on revient aux poids cibles dès que l'écart dépasse `rebalance_threshold`.

## Coûts (appliqués à chaque fill)

```
prix_fill = open × (1 ± (half_spread + slippage + impact) / 1e4)
impact_bps = impact_coef × sigma_barre × 1e4 × sqrt(notionnel / ADV_barre)
frais      = |qty| × prix_fill × fee_bps / 1e4
```

- `sigma_barre` et `ADV_barre` sont calculés sur les barres passées (décalés d'une barre), jamais sur la
  barre en cours.
- **Défauts :** taker 5 bps (Binance USDT-M VIP0), demi-spread 1 bp (0,5 pour BTC/ETH dans les scripts),
  slippage 1 bp, impact 0,5.
- **Spread dynamique** (`spread_model="volume"`, utilisé par la recherche) :
  `demi-spread = spread_coef_bps × sqrt(1e9 / volume journalier décalé)`, borné à [0,5 ; 25] bps.
  Cela donne environ 1 bp pour un marché à 1 Md$/jour, 3 bps à 100 M$ et 7 bps à 20 M$. Les altcoins
  illiquides paient donc beaucoup plus que BTC.
- **Cap de liquidité :** un fill ne dépasse pas `max_participation` (5 %) de l'ADV décalé. Le reste est
  exécuté sur les barres suivantes.
- **Stress tests :** `CostModel.scaled(k)`. Le funding n'est pas multiplié : c'est un flux de marché,
  pas un coût d'exécution.

## Exécution (`ExecutionModel`)

- **`taker` (défaut) :** ordre au marché à l'open de la barre suivante, avec les coûts ci-dessus.
- **`maker` :** ordre limite au close de décision, −1 bp pour un achat, +1 bp pour une vente
  (`limit_offset_bps`), re-coté à chaque barre tant qu'il n'est pas rempli.
  - **Fill :** au prix limite, uniquement si la barre **traverse** la limite de `trade_through_bps`
    (2 bps par défaut). C'est le haircut de position dans la file. Frais maker 2 bps perp, 10 bps spot.
  - **Ordre marketable :** si l'open est déjà au-delà de la limite (gap), fill taker à l'open.
  - **Antisélection :** incluse par construction, un achat n'est rempli que si le prix baisse à travers
    la limite.
  - **Réductions et sorties :** bascule en taker après `maker_fallback_bars` barres non remplies. Les
    entrées non remplies sont annulées si la cible change.
- **Validité à l'échelle de la barre :** aucune autre décision n'est prise pendant la barre. Seule la
  vérification de liquidation, déjà au pire extrême, partage la barre.
- **Limites :** pas de fills partiels liés à la file ; la latence de placement est supposée nulle.
- **Stress :** `BacktestConfig.stressed(k)` multiplie aussi les frais maker et le haircut de traversée.
- **Frais par symbole :** `per_symbol_fee_bps` et `per_symbol_maker_fee_bps` acceptent des règles de
  suffixe, par ex. `{"*@spot": 10.0}` pour les jambes spot.

## Open interest et positionnement (`metrics`)

- **Source :** archives quotidiennes Binance Vision, pas de vue mensuelle, échantillon toutes les
  5 minutes, disponibles depuis septembre 2020 (téléchargées depuis 2022).
- **Contenu :** open interest en base et en quote, ratio long/short des comptes retail, ratio
  long/short des positions des top traders, ratio de volume taker acheteur sur vendeur.
- **Alignement sur les barres :** la valeur d'une barre est la **dernière observation 5 minutes
  contenue dans la barre**, donc connue au close, comme tout autre signal. Les ratios de flux (taker)
  prennent la moyenne de la barre.
- **Nettoyage :** Binance publie parfois des open interest à zéro. Toute valeur non strictement
  positive est masquée au chargement, sinon les variations en pourcentage deviennent infinies et
  fabriquent des signaux.
- **Couverture :** les ratios top trader manquent dans environ 18 % des lignes, le ratio taker dans
  environ 7 %. Une stratégie qui les utilise doit tolérer ces trous.
- **Horodatage (vérifié contre l'API le 2026-10-06) :** les endpoints REST estampillent une valeur à
  la **fin** de sa période (documentation Binance : « End time of the period »), l'archive Vision
  estampille la même valeur au **début** de sa période de 5 minutes. Donc
  `create_time = t` dans l'archive ⟺ `timestamp = t + 5 min` côté REST. Un instantané estampillé `τ`
  côté REST est publié **12 à 138 secondes après `τ`** (mesuré).
- **Conséquence pour la recherche :** la barre `[T, T+Δ)` reçoit l'instantané pris **à sa clôture**
  (`τ = T+Δ`). Le backtest suppose donc un délai de publication nul. Effet mesuré : utiliser un
  instantané plus vieux d'une heure coûte 0,04 de Sharpe ; quelques minutes sont sans effet.
- **Conséquence pour le live :** tout ce qui vient du REST doit être regroupé sur `(T, T+Δ]`
  (`snapshots_to_bars`). Regrouper sur `[T, T+Δ)` comme pour l'archive donne à la stratégie des données
  **plus vieilles d'une barre** : Sharpe 1,16 → 0,86 sur la période de recherche. C'était le cas
  jusqu'au 2026-10-06.
- **Stockage :** les métriques sont conservées **rééchantillonnées en 1h**, en float32 compressé zstd
  (1,25 Go → 73 Mo pour 80 symboles). Les barres de recherche font 1h ou plus, donc rien n'est perdu ;
  pour du 5 minutes, il faut re-télécharger avec `--resample ''`.

## Funding (perps)

- Le taux publié à `calc_time` est rattaché à la barre qui contient cet instant. Plusieurs règlements
  dans une même barre sont additionnés.
- Paiement = `qty × prix_open × taux`, sur la position détenue **avant** les trades de la barre. Les
  longs paient si le taux est positif, les shorts reçoivent.
- Binance Vision publie le funding mensuellement. Au-delà du dernier mois publié, le funding est absent :
  il faut couper les backtests à la couverture du funding (les scripts utilisent `--end 2026-09-01`).

## Levier et liquidation

- Les poids sont plafonnés à `max_gross_leverage` (mise à l'échelle proportionnelle par ligne).
- **Liquidation (conservatrice) :** après les trades, l'equity est évaluée avec toutes les positions à
  leur extrême défavorable de la barre (low pour les longs, high pour les shorts), **simultanément**.
  Si elle passe sous `maintenance_margin × notionnel`, tout est liquidé à ces prix, avec les frais de
  liquidation. L'equity est plancher à 0 et le backtest s'arrête.
- Comptabilité : `equity = cash + Σ qty × prix`. Le cash peut être négatif (levier).

## Invariant comptable (testé)

`Σ PnL des trades (fermés + ouverts marqués) == equity finale − capital initial`, frictions comprises.
Seule exception : une liquidation où l'equity serait passée sous 0 (le fonds d'assurance absorbe).

## Métriques

- **Taux sans risque :** 0. **Année :** 365 jours (crypto 24/7).
- **Volatilité, Sharpe, Sortino :** calculés sur les rendements **journaliers** rééchantillonnés.
- **Max drawdown :** calculé sur tous les closes de barres. Le drawdown intrabarre, plus profond, n'est
  pas mesuré.
- **Statistiques de trades :** round trips fermés uniquement. Un trade dure tant que le signe de la
  position ne change pas. Un flip = deux trades.
- **Turnover annuel :** notionnel traité / equity moyenne / années.

## Limites connues (à traiter)

1. **Stops et limites intrabarre non modélisés.** Une stratégie à stop ou trailing stop ne doit pas être
   approximée ici. Elle attend le moteur événementiel (P4), qui fera une hypothèse de chemin défavorable.
2. **Trous de données :** réparés via l'API REST (`scripts/repair_data.py`) seulement si les barres REST
   adjacentes au trou sont identiques aux barres Vision. Les lignes réparées portent `source = 1`. Les
   trous de plus de 45 jours ne sont pas recousus (délisting ou réutilisation de ticker).
3. **Spread modélisé, pas mesuré :** le modèle en racine du volume n'est pas calibré sur le carnet
   d'ordres. Calibration à faire (bookTicker / aggTrades) avant le paper trading.
4. **Liquidation cross-margin approximée** (pire cas simultané) et pas de paliers de marge.
5. **Funding au prix open** au lieu du mark price.
