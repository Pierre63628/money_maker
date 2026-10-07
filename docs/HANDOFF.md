# Passation — money_maker

Document destiné à l'agent qui reprend le travail, en particulier sur le VPS. Il contient ce qui n'est
pas déductible du code : les décisions, les résultats, les pièges rencontrés et les règles à ne pas
enfreindre.

---

## 1. Objectif et garde-fous

Construire un système de trading algorithmique sur crypto, de la recherche à l'exécution, **sans jamais
se raconter d'histoires**. Le projet appartient à un utilisateur francophone : **répondre en français**.

**Règles absolues :**
- **Aucun ordre réel, aucune connexion à un compte réel, aucune dépense sans accord explicite.**
  `src/quant/brokers/live.py` et `mt5.py` lèvent une exception par conception. Ne pas les activer.
- **Le holdout est consommé** (voir §3). Il ne reste aucune donnée vierge. Toute amélioration doit être
  pré-enregistrée, testée sur 2022-2025 uniquement, puis validée en paper forward.
- **Ne jamais ajuster une stratégie après avoir vu un résultat hors échantillon.** C'est la règle que
  tout le protocole sert à protéger.
- **Falsifier les bons résultats.** Un Sharpe élevé est d'abord un bug jusqu'à preuve du contraire.

---

## 2. État du système

| Composant | État |
|---|---|
| Données | Binance USDT-M : barres 1h (682 symboles), funding, métriques de positionnement (414 symboles depuis 2021-12) |
| Moteur de backtest | Maker/taker, frais, spread calibré au volume, impact, funding, financement CFD, levier, remplissages partiels, délistings |
| Protocole | Walk-forward, Deflated Sharpe, PBO, bootstrap, stress coûts, régimes, registre, holdout |
| Recherche | 30 expériences, 159 essais, 11 familles. **Une seule stratégie survit** |
| Paper / testnet | Boucle complète, testée de bout en bout sur le testnet Binance |
| Livre directionnel | Paris quotidiens de Claude sur l'actualité, **paper**, prix mainnet ; une session par jour, aucune preuve d'edge à ce stade (`docs/DISCRETIONARY.md`) |
| Live | **Non implémenté et désactivé volontairement** |
| Tests | 178, hors ligne, dont des tests anti-anticipation et d'égalité recherche/live |

---

## 3. La stratégie retenue

**`CrowdingContrarian` / `CrowdingContrarianRobust`** (`src/quant/strategies/positioning.py`).

Binance publie le ratio long/short des **comptes particuliers** par perpétuel. On calcule son z-score
sur 90 jours, par symbole. On vend les perps où le retail est anormalement long, on achète ceux où il
est anormalement short. Quintiles de 20 %, rebalancement quotidien, portefeuille neutre au marché.

Le z-score est **relatif à l'historique du symbole**, pas à un seuil absolu : un token à 46 % de comptes
longs peut être shorté s'il tourne habituellement bien plus bas.

### Résultats

| Fenêtre | Rendement | Sharpe | MDD |
|---|---|---|---|
| Hors échantillon 2023-2025 (walk-forward, top-50 4h) | +223 % | 1,58 | −21 % |
| **Holdout 2026-01 → 2026-09 (consommé une seule fois)** | **+38,1 %** | 1,37 | −30,5 % |
| Variante robuste (vol inverse, plafond 5 %, âge min.) | +128 % | 1,22 | −19,6 % |

Pendant le holdout, BTC faisait −10,5 % et le bêta mesuré était −0,09 : c'est de l'alpha, pas une
exposition directionnelle déguisée.

### Tests de falsification (tous passés)

| Test | Résultat |
|---|---|
| Décalage des métriques de 4h / 8h / 24h | Sharpe 1,24 → 0,94 / 0,97 / 1,11 — pas un artefact d'horodatage |
| Même mécanique sur un z-score de **prix** | **−1,30** — ce n'est pas une réversion de prix déguisée |
| Ratios des **top traders** au lieu du retail | −0,04 à −0,57 — seul l'argent naïf porte le signal |
| 20 voisins de paramètres | 20 positifs, médiane 1,22, pire 0,91 |
| Jambe longue seule / courte seule | 0,56 et 0,32 — l'edge est dans **l'écart**, pas dans un côté |
| Rebalancement plus rapide (8h, 12h) | 0,66 et 0,94 — accélérer dégrade |
| Capacité | Sharpe 1,24 à 10 k$, 1,00 à 10 M$ |

### Les deux réserves, à ne jamais masquer

1. **Le Deflated Sharpe échoue** : 0,835 pour la variante robuste, seuil 0,9. Conséquence des 158 essais
   cumulés. **La stratégie n'est donc pas promue.**
2. **Les gains sont concentrés.** Sur le holdout, retirer les 3 meilleurs symboles fait passer +38 % à
   **−10,6 %**. Ces trois gagnants étaient des effondrements (−39 %, −57 %, −84 %) shortés pendant que
   le retail s'y entassait. **Le trade médian est légèrement négatif.** C'est une stratégie à queue
   épaisse : rentable sur la durée, irrégulière sur quelques mois, drawdowns d'environ 30 %.

---

### Ce que le live exécute, et à quoi le comparer (revue du 2026-10-06)

La boucle passe des ordres **au marché**. La recherche, elle, a été validée avec une exécution
**passive** (ordre limite au prix de décision, travaillé pendant la barre). Le forward se juge donc
contre le backtest *taker*, pas contre les chiffres ci-dessus :

| Fenêtre hors échantillon 2023-2025 | Sharpe | CAGR | MDD |
|---|---|---|---|
| Recherche validée (maker) | 1,58 | 48 % | −21 % |
| **Ce que la boucle exécute (taker)** | **1,34** | **39 %** | −22 % |

La revue a trouvé et corrigé un écart sérieux : jusqu'au 2026-10-06 le live lisait le positionnement
**une barre trop vieux** (Sharpe attendu 0,92 au lieu de 1,34). Voir §7 et `docs/RESEARCH_LOG.md`.
Les journaux antérieurs sont dans `var/paper/_archive/`.

---

## 4. Ce qui a été rejeté

Dix familles sur onze : momentum et trend cross-sectional (Sharpe 0,4-0,7, en déclin depuis 2024),
réversion horaire, RSI-2, lead-lag, saisonnalité, cascades de liquidation, momentum d'open interest,
multi-sleeve. Le cash-and-carry affiche un Sharpe de 4,6 mais ne rapporte que 7 % par an et reste
inactif quand le funding est bas — c'est le cas actuellement.

**Les stratégies importées (`research/`) étaient toutes des artefacts** : remplissages à des prix
impossibles, look-ahead, univers survivant. Voir `docs/RESEARCH_LOG.md`.

---

## 5. Architecture

```
src/quant/
  data/         Binance Vision + REST, store parquet, univers point-in-time, validation qualité
  backtesting/  moteur vectoriel (_core.py en numba), coûts, résultats
  strategies/   base, trend, cross_sectional, mean_reversion, events, seasonality, positioning, combine
  portfolio/    dimensionnement, maintien entre rebalancements
  risk/         moteur de limites, kill switch
  execution/    ordres, identifiants déterministes
  brokers/      base, paper, binance_futures (testnet), live (désactivé), mt5 (désactivé), profiles
  live/         feed (données live), loop (la boucle de trading)
  evaluation/   métriques, statistiques (DSR, PBO, bootstrap), validation, registre, audit de fills
  research/     expériences pré-enregistrées, runner, données d'univers
```

**Points d'entrée** : `scripts/research.py`, `download_data.py`, `download_metrics.py`,
`paper_trade.py`, `testnet_check.py`, `testnet_status.py`, `testnet_flatten.py`, `healthcheck.py`.

---

## 6. Le VPS

**Machine** : `ubuntu@<vps>` (adresse volontairement absente du dépôt), Ubuntu 26.04, Python 3.14.4, 2 vCPU, 3,8 Go + **4 Go de swap**,
38 Go de disque. Fuseau UTC, horloge synchronisée par chrony — **indispensable**, Binance rejette les
requêtes signées dont l'horodatage dérive.

**Tout appartient à `ubuntu`, aucun `sudo` nécessaire** pour travailler. Une règle polkit autorise
`systemctl` sur les deux services.

```
/opt/money_maker/           le projet
  .venv/                    environnement Python
  var/data/                 données de marché (~2 Go)
  var/secrets/              identifiants API, permissions 600, jamais versionnés
  var/paper/<id>/           journaux de boucle : events.jsonl, loop_state.json, broker/
  var/logs/                 sorties des services
```

**Services** : `quant-testnet` (vrais ordres sur le testnet Binance, argent fictif) et `quant-paper`
(exécutions simulées sur données réelles). Les deux redémarrent automatiquement, limités à 1 Go.
Activés au démarrage. Un cron lance `healthcheck.py` toutes les 30 minutes.

### Commandes

| Besoin | Commande (depuis `/opt/money_maker`) |
|---|---|
| État du compte et des positions | `.venv/bin/python scripts/testnet_status.py` |
| Santé des boucles | `.venv/bin/python scripts/healthcheck.py` |
| Écart modèle/réel | `.venv/bin/python scripts/paper_report.py var/paper/<id>` |
| **Tout solder en urgence** | `.venv/bin/python scripts/testnet_flatten.py` |
| Arrêt | `systemctl stop quant-testnet` |
| Kill switch sans arrêter le service | `touch var/paper/<id>/KILL` |
| Mise à jour du code | depuis le poste : `bash deploy/sync.sh ubuntu@<vps> --restart` |

---

## 7. Pièges rencontrés — ne pas les refaire

**Données**
- Binance publie des **barres gelées** (OHLC plat, volume nul) et des **open interest à zéro**. Non
  masqués, ils fabriquent des signaux. `clean_metrics` s'en charge.
- Les archives `metrics` sont **quotidiennes uniquement** et ne commencent qu'en **2021-12** pour la
  quasi-totalité des symboles (BTC depuis 2020-09). Aucun test sur période antérieure n'est possible.
- Les métriques sont stockées **rééchantillonnées en 1h, float32, zstd** : 1,25 Go → 73 Mo pour 80
  symboles. Stockées en 5 minutes, elles avaient saturé le disque du poste de travail.

**Horodatage du positionnement — le piège le plus coûteux trouvé**
- L'API REST estampille une valeur à la **fin** de sa période, l'archive Vision au **début** :
  `create_time = t` (archive) ⟺ `timestamp = t + 5 min` (REST). Publication 12 à 138 s après le tampon.
- La recherche donne à la barre `[T, T+Δ)` l'instantané de sa **clôture**. Tout ce qui vient du REST
  doit passer par `snapshots_to_bars` (regroupement `(T, T+Δ]`). Regrouper comme l'archive donne des
  données plus vieilles d'une barre, sans aucune erreur visible.
- `tests/test_metrics_alignment.py` prouve l'égalité recherche/live ; le journal de chaque pas contient
  `metrics_age_s`, qui doit rester à `[0, 0]` ou quelques minutes.

**Boucle et backtest doivent trader pareil**
- Le backtest ne trade un symbole que si **sa cible change** (`on_change`). La boucle fait de même et
  rattrape une décision manquée. `--rebalance drift` existe, testé, équivalent hors échantillon, non retenu.
- Un diagnostic favorable répété sur huit variantes d'une même période reste **une** observation : le
  maintien des poids semblait gagner 0,2 de Sharpe, le protocole a ramené l'écart à 0,06.

**Exchange**
- `0.0006 / 0.0001` vaut `5.999999…` en binaire : un arrondi plancher naïf retire un pas entier et fait
  passer l'ordre sous le notionnel minimum, **abandonné en silence**. D'où l'epsilon dans
  `round_quantity`.
- Le notionnel minimum **ne s'applique pas** aux ordres de réduction : l'appliquer quand même rend les
  positions résiduelles impossibles à fermer. Vérifié auprès de l'exchange.
- Le testnet ne liste **pas tout le mainnet** (45 symboles sur 50 de l'univers). `Broker.supported_symbols()`
  déclare ce que la venue accepte, la boucle restreint son univers en conséquence.

**Boucle**
- L'historique d'équité appartient à **un compte**. Repris d'un paper à 10 000 vers un testnet à 5 000,
  il affichait un faux drawdown de −50 % et bloquait tous les ordres. Chaque venue a son journal, et un
  changement de broker réinitialise l'historique avec une alerte.
- La boucle **refuse de trader une décision périmée** (plus de 45 minutes de retard). C'est voulu.

**Exploitation**
- Le poste de travail (macOS) **tue les processus de fond** par pression mémoire et s'endort : c'est la
  raison d'être du VPS.
- Le VPS à 3,8 Go **a fait tuer le téléchargement par le OOM killer** avec 8 symboles en parallèle.
  D'où le swap et les compteurs réduits (`BARS_WORKERS`, `METRIC_WORKERS`, `ARCHIVE_WORKERS`).
- Deux téléchargements concurrents s'étaient lancés et se disputaient la RAM : `bootstrap_data.sh`
  prend désormais un `flock`.
- macOS n'embarque qu'**openrsync** : `sync.sh` bascule sur tar si rsync échoue.
- **Attention aux `pkill -f`** dont le motif correspond à sa propre ligne de commande : j'ai coupé ma
  propre session SSH ainsi. Utiliser `[d]ownload` plutôt que `download`.

---

## 8. Ce qui reste à faire

0. **L'exécution passive**, premier levier d'efficacité et sans coût statistique : ordres limites
   post-only (`timeInForce=GTX`, rejet `-5022` si l'ordre croiserait) au prix de décision, re-cotés à
   chaque barre, repli au marché pour les réductions après trois barres. Gain modélisé : Sharpe 1,34 →
   1,58, CAGR 39 % → 48 %. Non implémenté ; touche au chemin des ordres réels.
1. **Le track record.** C'est le seul vrai manque. Laisser les deux boucles tourner plusieurs semaines
   sans interruption, puis comparer avec `paper_report.py` : l'écart d'exécution observé contre le
   modèle, et la performance contre l'attendu (fenêtres de 8 mois rééchantillonnées : médiane +27,6 %,
   p5 −8,3 %, probabilité de perte 11 %).
2. **Le dépôt git n'a aucun commit.** À faire.
3. **Améliorations possibles**, à pré-enregistrer avant de les tester : frein de drawdown, exclusion des
   cotations très récentes, pondération plus fine. Toute idée nouvelle augmente N et durcit la porte DSR.
4. **Passage en réel** : non décidé. Exigerait l'accord explicite du propriétaire, l'implémentation du
   broker mainnet avec réconciliation et kill switch éprouvés, et **au minimum 1 500 $** — en dessous,
   le pas de lot de BTC (0,001, environ 86 $) rend le portefeuille non reproductible.

---

## 9. Documents à lire

| Fichier | Contenu |
|---|---|
| `docs/RESEARCH_LOG.md` | Journal complet : chaque batch, chaque rejet, chaque raison |
| `docs/DISCRETIONARY.md` | Livre directionnel : limites, sorties rejouées, test contre le hasard, porte, journal des sessions |
| `docs/BACKTEST_CONVENTIONS.md` | Conventions de timing, coûts, pièges de données |
| `docs/ARCHITECTURE.md` | Carte des modules et phases |
| `deploy/README.md` | Procédure de déploiement et d'exploitation |
| `var/experiments/registry.jsonl` | Les 158 essais, horodatés, avec leurs métriques |
| `var/reports/research/<exp>/report.md` | Un rapport par expérience, portes détaillées |
