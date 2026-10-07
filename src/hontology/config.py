"""Process configuration.

The split enforced here is the one the whole project depends on: **behavior** is
hashed into a run's identity, **infrastructure** is not.

A provider name and a model name change what comes out of the pipeline, so two
runs that differ in them are two different experiments. A host, a port, a
connection URL or an API key is a transport detail — the same run executed
against a model served from a different machine is the *same* run. Infrastructure
therefore lives here, in process settings, and is recorded in a run manifest for
provenance; it never enters a config hash.

Getting that backwards makes every host move look like a new experiment and
quietly destroys the ability to compare results across machines.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HONTOLOGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage (infrastructure) -------------------------------------------
    database_url: str = "postgresql+psycopg://hontology:hontology@localhost:5435/hontology"
    data_dir: Path = Path("./data")

    # --- API (infrastructure) -----------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8100
    api_base_url: str = "http://127.0.0.1:8100"

    # --- Provider transports (infrastructure) -------------------------------
    ollama_host: str = "http://localhost:11434"
    # llama-server holds one model per process, so a chat model and an embedding
    # model are usually two servers. The embed host falls back to the chat host.
    llamacpp_host: str = "http://localhost:8080"
    llamacpp_embed_host: str | None = None
    llamacpp_api_key: str | None = None
    # Hosted models through OpenRouter. The key never belongs in a config.
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Provider defaults (behavior: hashed when a run adopts them) --------
    default_judge_provider: str = "ollama"
    default_judge_model: str = "huihui_ai/qwen3.5-abliterated:9b"
    default_embed_provider: str = "ollama"
    default_embed_model: str = "nomic-embed-text"

    # --- Ingest --------------------------------------------------------------
    gdelt_base_url: str = "https://data.gdeltproject.org/gdeltv2"
    scrape_user_agent: str = "hontology/0.1 (+https://github.com/emaroppo/hontology)"
    scrape_max_concurrency: int = 8
    scrape_per_host_delay_s: float = 1.0
    scrape_timeout_s: float = 20.0
    scrape_budget: int = 500

    # --- Derived paths -------------------------------------------------------
    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def scrape_cache_dir(self) -> Path:
        return self.data_dir / "scrape_cache"

    @property
    def feed_cache_dir(self) -> Path:
        return self.data_dir / "feeds"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.scrape_cache_dir,
            self.feed_cache_dir,
            self.snapshots_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def psycopg_url(self) -> str:
        """The same URL in the form the raw psycopg driver accepts."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
