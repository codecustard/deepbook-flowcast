import os
import sys

import typer

app = typer.Typer(help="deepbook-flowcast: when does ML beat the imbalance rule on DeepBook?",
                  no_args_is_help=True)


@app.callback()
def _line_buffered() -> None:
    """Print progress line by line even when piped (e.g. into `tee`), including from
    parallel worker processes (they inherit PYTHONUNBUFFERED)."""
    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


# ------------------------------------------------------------ collector (raw data only)

@app.command()
def collect(
    start: str = typer.Option(..., help="first day, e.g. 2025-08-01 (UTC)"),
    end: str = typer.Option(..., help="day AFTER the last one, e.g. 2025-11-01 (exclusive)"),
    pools: str = typer.Option(None, help="comma-separated, default from .env POOLS"),
    streams: str = typer.Option("orders,trades", help="orders, trades, or both"),
    workers: int = typer.Option(None, help="parallel requests, default from .env WORKERS"),
    redo: bool = typer.Option(False, help="re-fetch hours already on disk"),
):
    """[collector] Backfill raw order updates and trades -> data/raw/ (resumable)."""
    from flowcast.collector import collect as c
    from flowcast.config import settings

    c.run(pools=pools.split(",") if pools else settings.pool_list, streams=streams.split(","),
          start=c.parse_day(start), end=c.parse_day(end), workers=workers, redo=redo)


@app.command()
def binance(
    start: str = typer.Option(..., help="first day, e.g. 2025-08-01 (UTC)"),
    end: str = typer.Option(..., help="day AFTER the last one (exclusive)"),
    symbols: str = typer.Option(None, help="e.g. SUIUSDC,um:DEEPUSDT; default from .env BINANCE_SYMBOLS"),
    workers: int = typer.Option(None, help="parallel downloads, default from .env WORKERS"),
    redo: bool = typer.Option(False, help="re-fetch days already on disk"),
):
    """[collector] Binance aggTrades for the lead-lag angle -> data/raw/binance/ (resumable)."""
    from flowcast.collector import binance as b
    from flowcast.collector.collect import parse_day
    from flowcast.config import settings

    b.run(symbols.split(",") if symbols else settings.binance_list,
          parse_day(start).date(), parse_day(end).date(), workers=workers, redo=redo)


@app.command()
def status():
    """[collector] What's on disk: DeepBook hours and Binance days, gaps, rows."""
    from flowcast.collector import binance as b
    from flowcast.collector import collect as c

    c.status()
    b.status()


@app.command()
def snapshots(
    pools: str = typer.Option(None, help="comma-separated, default from .env POOLS"),
    every: float = typer.Option(2.0, help="seconds between polls"),
    depth: int = typer.Option(20, help="levels per side"),
):
    """[collector] Record live order book snapshots -> data/snapshots/ (runs until Ctrl-C)."""
    from flowcast.collector import snapshots as s
    from flowcast.config import settings

    s.record(pools.split(",") if pools else settings.pool_list, every=every, depth=depth)


@app.command()
def pools():
    """[collector] List DeepBook pools the indexer knows about."""
    from flowcast.collector.indexer_api import DeepBookIndexer

    for p in DeepBookIndexer().pools():
        print(p.get("pool_name"))


# ------------------------------------------------------------ ML side (reads data/raw, never writes it)

@app.command()
def book(pool: str = "SUI_USDC", start: str = "2025-08-01", end: str = "2025-11-01"):
    """Rebuild the order book from raw events (week 2)."""
    from flowcast import book as b

    b.rebuild(pool, start, end)


@app.command()
def labels(pools: str = "SUI_USDC,DEEP_USDC"):
    """Up/flat/down labels at 1s/10s/60s from rebuilt books (thresholds fit on Aug 2025)."""
    from flowcast.labels import make_labels

    for p in pools.split(","):
        make_labels(p)


@app.command()
def baselines(pools: str = "SUI_USDC,DEEP_USDC"):
    """Score flat / imbalance / Binance lead-lag rules: fit on Aug 2025, scored on Sep 2025."""
    from flowcast.models import rules

    for p in pools.split(","):
        rules.run(p)


@app.command()
def features(pools: str = "SUI_USDC,DEEP_USDC",
             redo: bool = typer.Option(False, help="rebuild days already on disk")):
    """Per-second features (X) from rebuilt books, raw events and Binance -> data/processed/features/."""
    from flowcast import features as f

    for p in pools.split(","):
        f.build(p, redo=redo)


@app.command()
def train(kind: str = typer.Argument("logreg", help="model kind: logreg or lgbm"),
          pools: str = "SUI_USDC,DEEP_USDC", horizons: str = "1s,10s,60s"):
    """Train on Aug 2025 (tuned with TimeSeriesSplit), score on Sep 2025 next to the best rule."""
    from flowcast.models import train as t

    t.run(kind, pools=tuple(pools.split(",")), horizons=tuple(horizons.split(",")))


@app.command()
def evaluate(pools: str = "SUI_USDC,DEEP_USDC"):
    """Bootstrap intervals, paired comparisons, confident-call economics and lead-lag on Sep 2025."""
    from flowcast import evaluate as e

    e.run(tuple(pools.split(",")))


@app.command()
def walkforward(pools: str = "SUI_USDC,DEEP_USDC", horizons: str = "1s,10s,60s"):
    """Walk-forward over September 2025 weeks: retrained LightGBM vs fixed Aug model vs best rule."""
    from flowcast.models import walkforward as wf

    wf.run(tuple(pools.split(",")), tuple(horizons.split(",")))


@app.command(name="final-train")
def final_train(pools: str = "SUI_USDC,DEEP_USDC"):
    """Retrain LightGBM on Aug 2 - Sep 30 (frozen settings) and refit the rules, for the held-out tests."""
    from flowcast.models import holdout as ho

    ho.train_final(tuple(pools.split(",")))


@app.command()
def holdout(period: str = typer.Argument(..., help="oct2025 or 2026"),
            final_run: str = typer.Option(None, help="runs/ folder from final-train; trained now if omitted"),
            force: bool = typer.Option(False, help="re-run a period already tested (record why in LOG.md)")):
    """One-time held-out test: primary model vs alternative vs best rule, with bootstrap intervals."""
    from flowcast.models import holdout as ho

    ho.run(period, final_run=final_run, force=force)


if __name__ == "__main__":
    app()
