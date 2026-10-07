"""The numbers a decision is allowed to rest on: the state of the book and of the tradable market.

The brief is produced by code and saved next to the decision it informed, so that what was known
when a call was made can be read again later. Prices, funding and positioning come from the venue,
never from memory or from a web page.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from quant.discretionary.book import DiscretionaryBook
from quant.discretionary.scoring import scorecard


def market_table(raw: dict[str, pd.DataFrame], mids: pd.Series, volume: pd.Series, funding: pd.Series,
                 positioning: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for symbol, bars in raw.items():
        close = bars["close"]
        if close.empty:
            continue
        last = float(mids.get(symbol, close.iloc[-1]))

        def change(hours: int) -> float:
            # a bar is indexed by its open: the close known ``hours`` ago belongs to the bar opened an hour earlier
            ref = close.asof(now - pd.Timedelta(hours=hours + 1))
            return last / ref - 1 if pd.notna(ref) else np.nan

        daily = close.resample("1D").last().pct_change(fill_method=None)
        rows[symbol] = {"prix": last, "24h": change(24), "7j": change(24 * 7), "30j": change(24 * 30),
                        "vol_jour": float(daily.iloc[-30:].std()), "funding": float(funding.get(symbol, np.nan)),
                        "volume_24h_M": float(volume.get(symbol, np.nan)) / 1e6}
    table = pd.DataFrame.from_dict(rows, orient="index")
    if len(positioning):
        table = table.join(positioning.rename(columns={"ratio": "LS_retail", "z": "LS_z"}))
    return table.sort_values("volume_24h_M", ascending=False)


def build_brief(book: DiscretionaryBook, now: pd.Timestamp, refreshed: dict, positioning: bool = True) -> dict:
    """``refreshed`` is the result of ``book.refresh(now)``, run by the caller just before."""
    whitelist = book.whitelist()
    held = [c.symbol for c in book.open_calls()]
    symbols = list(dict.fromkeys(whitelist + held))
    raw = book.hourly_bars(symbols, now)
    _, marks = book.marks(symbols)
    try:
        funding = book.feed.funding_now()
    except Exception:  # context only: the brief is still usable without it
        funding = pd.Series(dtype=float)
    crowd = pd.DataFrame()
    if positioning:
        try:
            crowd = book.feed.retail_positioning(symbols, now)
        except Exception:
            pass
    open_rows = []
    for c in book.open_calls():
        mark = float(marks[c.symbol])
        move = c.sign * (mark / c.entry_price - 1)
        open_rows.append({
            "call_id": c.call_id, "sens": c.direction, "poids": c.weight, "entrée": c.entry_price, "marque": mark,
            "latent": move, "R": move / c.stop_pct, "stop": c.entry_price * (1 - c.sign * c.stop_pct),
            "objectif": c.entry_price * (1 + c.sign * c.target_pct) if c.target_pct else np.nan,
            "jours_restants": (pd.Timestamp(c.expiry) - now) / pd.Timedelta(days=1),
            "thèse": c.thesis, "invalidation": c.invalidation})
    lim = book.cfg.limits
    return {
        "now": now, "refreshed": refreshed, "whitelist": whitelist, "open": open_rows,
        "market": market_table(raw, marks, book.feed.quote_volume_24h(), funding, crowd, now),
        "score": scorecard(book.state.calls.values(), book.equity_points()),
        "decision_today": None if book.committed_without_calls(f"{now:%Y-%m-%d}")
        else book.state.decisions.get(f"{now:%Y-%m-%d}"),
        "room": {"slots": lim.max_open_calls - len(open_rows),
                 "gross": book.cfg.risk_limits.max_gross_leverage - refreshed["gross"],
                 "net_limit": book.cfg.risk_limits.max_net_leverage},
        "limits": asdict(lim), "risk_limits": asdict(book.cfg.risk_limits),
    }


def render_brief(brief: dict) -> str:
    now, state, score = brief["now"], brief["refreshed"], brief["score"]
    pct = "{:+.2%}".format
    lines = [f"BRIEF DU LIVRE DIRECTIONNEL, {now:%Y-%m-%d %H:%M} UTC (paper, prix mainnet Binance USDT-M)", ""]
    if state["killed"]:
        lines += [f"!! KILL SWITCH ACTIF : {state['kill_reason']}",
                  "   Aucun nouveau pari tant que le propriétaire ne l'a pas réarmé (reset-kill).", ""]
    lines.append(f"Équité {state['equity']:,.2f} USDT | brut {state['gross']:.1%} | net {state['net']:+.1%} | "
                 f"{state['open_calls']} paris ouverts | BTC {state['benchmark']:,.1f}")
    if score["closed_calls"]:
        lines.append(f"Historique : {score['closed_calls']} paris clos, réussite {score['hit_rate']:.0%}, "
                     f"rendement {score['total_return']:+.2%}, excès sur le hasard {score['edge']:+.2%} "
                     f"(p = {score['p_value']:.2f}, sur {score['scored_calls']} paris comparés), "
                     f"Brier {score['brier']:.3f}")
    if brief["decision_today"]:
        lines.append(f"!! Une décision a déjà été exécutée aujourd'hui ({brief['decision_today']['executed_at'][:16]}) : "
                     "elle est immuable, pas de seconde décision avant demain (UTC).")

    lines += ["", "SORTIES DEPUIS LA DERNIÈRE SESSION"]
    lines += [f"  {e['call_id']} {e['direction']} : {e['exit_reason']} à {e['exit_price']:.6g} le "
              f"{e['exit_time'][:16]}, {e['net_return']:+.2%} du notionnel ({e['r_multiple']:+.2f} R), "
              f"{e['pnl']:+.2f} USDT" for e in state["exits"]] or ["  aucune"]

    lines += ["", "PARIS OUVERTS"]
    for row in brief["open"]:
        target = f"{row['objectif']:.6g}" if pd.notna(row["objectif"]) else "aucun"
        lines += [f"  {row['call_id']} {row['sens'].upper()} {row['poids']:.1%} | entrée {row['entrée']:.6g}, marque "
                  f"{row['marque']:.6g}, latent {row['latent']:+.2%} ({row['R']:+.2f} R) | stop {row['stop']:.6g}, "
                  f"objectif {target} | reste {row['jours_restants']:.1f} j",
                  f"      thèse : {row['thèse']}", f"      invalidation : {row['invalidation']}"]
    if not brief["open"]:
        lines.append("  aucun")

    room, lim, risk = brief["room"], brief["limits"], brief["risk_limits"]
    lines += ["", "MARGE DE MANOEUVRE ET LIMITES (dans le code, non négociables)",
              f"  {room['slots']} emplacements libres sur {lim['max_open_calls']} | au plus "
              f"{lim['max_new_calls_per_day']} nouveaux paris par jour | brut disponible {room['gross']:.1%} "
              f"(plafond {risk['max_gross_leverage']:.0%}), net plafonné à {room['net_limit']:.0%}",
              f"  par pari : poids ≤ {lim['max_position_weight']:.0%}, poids x stop ≤ {lim['max_risk_per_call']:.1%} "
              f"de l'équité, stop entre {lim['min_stop_pct']:.0%} et {lim['max_stop_pct']:.0%}, horizon ≤ "
              f"{lim['max_horizon_days']} j, confiance entre {lim['min_confidence']:.0%} et {lim['max_confidence']:.0%}",
              f"  livre : perte du jour {-risk['max_daily_loss']:.0%} = plus de nouveau risque ; drawdown "
              f"{-risk['max_drawdown']:.0%} = tout est soldé et le livre s'arrête"]

    market = brief["market"]
    formats = {"prix": "{:.6g}".format, "24h": pct, "7j": pct, "30j": pct, "vol_jour": "{:.1%}".format,
               "funding": "{:+.4%}".format, "volume_24h_M": "{:,.0f}".format, "LS_retail": "{:.2f}".format,
               "LS_z": "{:+.1f}".format}
    lines += ["", f"MARCHÉ : les {len(brief['whitelist'])} perpétuels autorisés aujourd'hui (plus les positions ouvertes)",
              "  funding = dernier taux par intervalle de règlement ; LS_retail = ratio long/short des comptes, "
              "LS_z = son écart à 25 jours",
              market.to_string(formatters={k: v for k, v in formats.items() if k in market}, na_rep="-")]
    return "\n".join(lines)
