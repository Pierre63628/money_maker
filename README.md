# money_maker : moteur de recherche quantitative et de trading algorithmique

Objectif : trouver des edges court terme **réels** (après coûts, hors échantillon), les valider, les
paper-trader, puis seulement ensuite envisager le live, **avec approbation manuelle**.

> Aucun rendement n'est garanti. Un bon backtest n'est pas une preuve de rentabilité future.
> Aucun ordre réel n'est passé et aucun broker réel n'est connecté sans autorisation explicite.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # 113 tests, hors-ligne
```

## Utilisation

```bash
# 1. Données : Binance USDT-M, checksums vérifiés, rapport qualité, funding
.venv/bin/python scripts/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --intervals 1h 15m --start 2020-01-01

# 2. Baselines + benchmarks + stress coûts (x0, x1, x2)
.venv/bin/python scripts/run_baselines.py --interval 1h --end 2026-09-01

# 3. Audit des stratégies Strategy Factory (artefacts de fills)
.venv/bin/python scripts/audit_strategy_factory.py --sort sharpe --limit 15

# 4. Univers complet (679 perps crypto, délistés inclus) + réparation des trous via REST
.venv/bin/python scripts/download_data.py --universe crypto --intervals 1h --workers 10 --skip-existing
.venv/bin/python scripts/repair_data.py --interval 1h --all --no-funding

# 4b. Open interest et positionnement (archives quotidiennes, ~4 Go pour le top-30 depuis 2022)
.venv/bin/python scripts/download_metrics.py --top-n 30 --start 2022-01-01 --workers 8

# 5. Recherche pré-enregistrée : walk-forward, DSR, PBO, bootstrap, régimes, portes de promotion
.venv/bin/python scripts/research.py --list
.venv/bin/python scripts/research.py trend_top10_4h multisleeve_top30_4h
#    rapports : var/reports/research/<expérience>/report.md ; registre : var/experiments/registry.jsonl

# 5b. Faisabilité chez un courtier (CFD/MT5) : coûts réels, instruments listés, spread d'équilibre
.venv/bin/python scripts/broker_feasibility.py --profile mon_courtier.json

# 6. Paper trading (données publiques, AUCUN compte, AUCUN ordre réel)
.venv/bin/python scripts/paper_trade.py --experiment multisleeve_top30_4h --once --shadow
.venv/bin/python scripts/paper_report.py var/paper/multisleeve_top30_4h-c0-shadow
touch var/paper/multisleeve_top30_4h-c0-shadow/KILL     # arrêt d'urgence : liquide les positions paper
```

En Python :

```python
from quant.data.store import ParquetStore
from quant.data.market_data import load_market_data
from quant.backtesting.engine import BacktestConfig, run_backtest
from quant.evaluation.lookahead import check_no_lookahead
from quant.strategies.baselines import TimeSeriesMomentum

data = load_market_data(ParquetStore("var/data"), ["BTCUSDT", "ETHUSDT"], "1h", "2021-01-01", "2026-09-01")
strategy = TimeSeriesMomentum(lookback=720)
assert check_no_lookahead(strategy, data).passed
result = run_backtest(strategy.target_weights(data), data, BacktestConfig(), name=strategy.name)
print(result.metrics())
```

## Documentation

- `docs/ARCHITECTURE.md` : architecture, état des modules, roadmap, registre des risques
- `docs/BACKTEST_CONVENTIONS.md` : contrat du backtester (timing, coûts, funding, liquidation, métriques)
- `docs/STRATEGY_FACTORY.md` : analyse et audit de Strategy Factory, règles de provenance
- `docs/RESEARCH_LOG.md` : journal des essais et résultats

## Structure

```
src/quant/{data,features,signals,strategies,risk,portfolio,backtesting,execution,brokers,models,evaluation,monitoring,integrations}
scripts/   CLI
tests/     pytest (synthétique, hors-ligne)
var/       données et rapports (non versionnés)
```
