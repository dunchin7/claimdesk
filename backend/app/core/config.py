from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the workspace root (one level above backend/)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings.

    NOTE: We use TWO Azure OpenAI resources — one for chat completions, one
    for embeddings. They have different endpoints, keys, and API versions.
    The LLM abstraction (`app/ai/llm.py`) routes per model-alias to the right
    resource.
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

    # --- Azure OpenAI: chat resource ---
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    # Single chat deployment for now. Reasoner/extractor/vision/judge all
    # alias to this until a separate gpt-4o deployment exists.
    azure_openai_deployment: str = "gpt-4o-mini"

    # --- Azure OpenAI: embeddings resource (separate Azure account) ---
    azure_openai_embedding_api_key: str = ""
    azure_openai_embedding_endpoint: str = ""
    azure_openai_embedding_api_version: str = "2023-05-15"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    # ada-002 = 1536; 3-small = 1536; 3-large = 3072. Used by pgvector schema.
    azure_openai_embedding_dim: int = 1536

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
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def embedding_configured(self) -> bool:
        return bool(
            self.azure_openai_embedding_api_key
            and self.azure_openai_embedding_endpoint
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
