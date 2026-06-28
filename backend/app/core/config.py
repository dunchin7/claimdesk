from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the workspace root (one level above backend/)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings.

    The LLM abstraction (`app/ai/llm.py`) reads the OpenAI settings below and
    routes every model-alias through a single provider. Point `openai_base_url`
    at any OpenAI-compatible endpoint to swap providers without code changes.
    """

    model_config = SettingsConfigDict(
        env_file=WORKSPACE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_env: Literal["dev", "staging", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- OpenAI ---
    # One key for chat + embeddings. Set `openai_base_url` to use any
    # OpenAI-compatible endpoint (a gateway, a local server, etc.).
    openai_api_key: str = ""
    openai_base_url: str = ""  # blank = https://api.openai.com/v1
    # Cheap tier — extraction, vision, and email prose.
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # text-embedding-3-small = 1536; ada-002 = 1536; 3-large = 3072.
    # Drives the pgvector column dimension (see migrations).
    embedding_dim: int = 1536

    # Frontier tier — the hard reasoning steps (adjudication + the citation
    # verification judge). Blank fields fall back to the chat model / key /
    # base above, so the default config is single-model and nothing changes.
    # Set these to route the `reasoner` and `judge` aliases to a stronger
    # model — optionally a *different provider* (give it its own key + base;
    # LiteLLM handles the routing). Extraction and vision stay on the cheap tier.
    openai_reasoner_model: str = ""
    openai_reasoner_api_key: str = ""
    openai_reasoner_base_url: str = ""

    # --- Local embeddings (Week 6 BGE comparison) ---
    # BAAI/bge-large-en-v1.5 is 1024-dim. Other BGE variants:
    #   bge-base-en-v1.5  → 768
    #   bge-small-en-v1.5 → 384
    bge_model_name: str = "BAAI/bge-large-en-v1.5"
    bge_embedding_dim: int = 1024

    # --- Database ---
    database_url: str = "postgresql+asyncpg://dev:dev@localhost:5432/claims_copilot"
    alembic_database_url: str = "postgresql+psycopg://dev:dev@localhost:5432/claims_copilot"

    # --- Langfuse ---
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- Future weeks ---
    cohere_api_key: str = ""
    sentry_dsn: str = ""
    resend_api_key: str = ""
    resend_webhook_secret: str = ""
    shopify_api_key: str = ""
    shopify_api_secret: str = ""
    shopify_store_domain: str = ""

    # Convenience: per-claim cost cap (USD), enforced by Week 11+ agents.
    per_claim_cost_cap_usd: float = Field(default=0.50, ge=0.0)

    # --- Phase-4 P4.2 safety override ---
    # The XGBoost calibrator (v3) is trained on synthetic data only. The
    # 2026-06-06 real-data mini-eval showed it auto-resolves real claims
    # at extreme confidence (median 0.994) but is right only 1/8 of the
    # time on labeled rows. Until we retrain on operator-verified real
    # claims, production deployments MUST set this to true — the pipeline
    # will then demote every `auto_resolve` route to `review` with the
    # explicit reason `auto_resolve_disabled_synthetic_calibrator` so
    # operators see every claim.
    #
    # Dev / CI keep the default (False) so the synthetic regression suite
    # still measures prompt and calibrator iterations honestly.
    disable_auto_resolve: bool = False

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def chat_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
