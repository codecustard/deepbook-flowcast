from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"              # collector output: raw indexer events, append-only
SNAPSHOTS = DATA / "snapshots"  # collector output: live order book snapshots
PROCESSED = DATA / "processed"  # ML side: rebuilt books, features, labels
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    indexer_base: str = "https://deepbook-indexer.mainnet.mystenlabs.com"
    # SUI_USDC = main pool; DEEP_USDC = thin pool, test-only (does the result carry over?)
    pools: str = "SUI_USDC,DEEP_USDC"
    # Binance reference prices for the lead-lag angle. DEEP has no Binance spot market,
    # so it uses the USD-M perpetual ("um:").
    binance_symbols: str = "SUIUSDC,SUIUSDT,um:DEEPUSDT"
    workers: int = 4
    request_timeout: float = 60.0

    @property
    def pool_list(self) -> list[str]:
        return [p.strip() for p in self.pools.split(",") if p.strip()]

    @property
    def binance_list(self) -> list[str]:
        return [s.strip() for s in self.binance_symbols.split(",") if s.strip()]


settings = Settings()
