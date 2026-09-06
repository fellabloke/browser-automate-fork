"""Provider SDK adapters and model pipeline construction.

Construction is intentionally separate from routing, health, and failover policy.
The exported builders preserve the registry's existing ordering and metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from .schemas import ModelClient


try:
    from agent_first_browse.logging import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


def _collect_keys(*env_vars: str) -> list[str]:
    """Collect all unique API keys from multiple env vars (comma-separated)."""
    keys: list[str] = []
    for var in env_vars:
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        for key in raw.replace(";", ",").split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _credential_fingerprint(secret: str) -> str:
    """Stable, non-reversible credential identity for ModelClient metadata."""
    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()[:16]


def _premium_config() -> dict:
    """Premium single-key/model config — one paid key that does text + vision."""
    return {
        "keys": _collect_keys("PREMIUM_API_KEY", "PREMIUM_API_KEYS"),
        "model": os.getenv("PREMIUM_MODEL", "").strip(),
        "vision_model": os.getenv("PREMIUM_VISION_MODEL", "").strip(),
        "base_url": os.getenv("PREMIUM_BASE_URL", "https://api.openai.com/v1").strip(),
        "provider": os.getenv("PREMIUM_PROVIDER", "openai").strip().lower(),
        "timeout": int(os.getenv("PREMIUM_TIMEOUT", "60")),
    }


def _extract_json_payload(text: str) -> dict:
    from .failover import extract_json_payload

    return extract_json_payload(text)


def _compact_provider_error(error: str, limit: int = 280) -> str:
    from .failover import _compact_provider_error

    return _compact_provider_error(error, limit)


class CloudflareNativeVisionClient:
    """LangChain-compatible adapter for Workers AI's native vision route.

    Cloudflare's OpenAI-compatible endpoint serves the configured text models,
    but Llama 3.2 vision currently requires the native ``/ai/run/<model>``
    response envelope. This keeps that transport quirk inside the provider adapter.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        model: str,
        timeout: float = 45.0,
        schema: type | None = None,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.model_name = model
        self.timeout = timeout
        self._schema = schema

    def with_structured_output(self, schema: type):
        return CloudflareNativeVisionClient(
            account_id=self.account_id,
            api_token=self.api_token,
            model=self.model_name,
            timeout=self.timeout,
            schema=schema,
        )

    @staticmethod
    def _message_payload(message: Any) -> dict[str, Any]:
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }.get(getattr(message, "type", ""), getattr(message, "role", "user"))
        return {"role": role, "content": getattr(message, "content", str(message))}

    async def ainvoke(self, messages: list, config: Any = None) -> Any:
        import httpx
        from langchain_core.messages import AIMessage

        payload_messages = [self._message_payload(message) for message in messages]
        if self._schema is not None:
            schema_json = json.dumps(self._schema.model_json_schema(), ensure_ascii=False)
            payload_messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Return ONLY one valid JSON object matching this JSON Schema. "
                        "Do not use markdown fences or add commentary.\n"
                        f"SCHEMA: {schema_json}"
                    ),
                },
            )

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model_name}"
        )
        request_payload: dict[str, Any] = {
            "messages": payload_messages,
            "max_tokens": _int_env("VISION_MAX_TOKENS", 1000, 256),
            "temperature": 0.0,
        }
        if self._schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": self._schema.model_json_schema(),
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cloudflare Workers AI HTTP {response.status_code}: invalid JSON response"
            ) from exc

        if response.status_code >= 400 or not body.get("success", False):
            errors = body.get("errors") or []
            detail = errors[0].get("message", "request failed") if errors else "request failed"
            raise RuntimeError(
                f"Cloudflare Workers AI HTTP {response.status_code}: "
                f"{_compact_provider_error(str(detail))}"
            )

        result = body.get("result") or {}
        content = result.get("response", "")
        if self._schema is not None:
            try:
                payload = content if isinstance(content, dict) else _extract_json_payload(content)
                return self._schema.model_validate(payload)
            except Exception:  # noqa: BLE001
                # Llama Vision occasionally ignores JSON mode when an image is
                # present even though Cloudflare lists it as supported. Repair
                # the already-grounded natural-language observation with one
                # cheap text-only turn on the same model (no image reprocessing).
                logger.debug(
                    "Cloudflare vision returned non-JSON; running text-only schema repair"
                )
                repair_payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return ONLY one valid JSON object matching this JSON Schema. "
                                "No prose or markdown.\n"
                                f"SCHEMA: {json.dumps(self._schema.model_json_schema(), ensure_ascii=False)}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Convert this visual observation into the required JSON without "
                                "adding unsupported claims:\n" + str(content)
                            ),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": self._schema.model_json_schema(),
                    },
                    "max_tokens": _int_env("VISION_MAX_TOKENS", 1000, 256),
                    "temperature": 0.0,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    repair_response = await client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
                            "Content-Type": "application/json",
                        },
                        json=repair_payload,
                    )
                repair_body = repair_response.json()
                if repair_response.status_code >= 400 or not repair_body.get("success", False):
                    repair_errors = repair_body.get("errors") or []
                    repair_detail = (
                        repair_errors[0].get("message", "JSON repair failed")
                        if repair_errors else "JSON repair failed"
                    )
                    raise RuntimeError(
                        f"Cloudflare Workers AI HTTP {repair_response.status_code}: "
                        f"{_compact_provider_error(str(repair_detail))}"
                    )
                repaired = (repair_body.get("result") or {}).get("response", "")
                repaired_payload = (
                    repaired if isinstance(repaired, dict) else _extract_json_payload(repaired)
                )
                return self._schema.model_validate(repaired_payload)
        return AIMessage(
            content=content if isinstance(content, str) else json.dumps(content),
            response_metadata={"usage": result.get("usage", {})},
        )


# ═══════════════════════════════════════════════════════════════════════════════
def _build_text_pipeline() -> list[ModelClient]:
    """TEXT-ONLY pipeline for Manager & Helper. NEVER receives images.

    Multi-model-per-key architecture:
      On NVIDIA NIM, ONE API key gives access to ALL models. So we register
      multiple models under the same key. The failover loop tries every model
      on each key before moving to the next provider.
    """
    from langchain_openai import ChatOpenAI

    clients: list[ModelClient] = []
    # Structured browser actions are compact; cap completion length to keep
    # provider requests bounded.
    text_max_tokens = _int_env("TEXT_MODEL_MAX_TOKENS", 1000, 256)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }

    # ── NVIDIA NIM — SECONDARY (multiple models, same key) ──
    # Register the same top-tier model (gpt-oss-120b) here too, so the
    # model-first failover can continue with the identical model on NVIDIA when
    # all Groq keys are rate-limited. If the catalog ID is invalid on NVIDIA,
    # the startup probe prunes it automatically — zero per-step cost.
    # Override via NVIDIA_TEXT_MODELS="model1,model2" in .env.
    nvidia_keys = _collect_keys("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS")
    base = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    _nvidia_models_raw = os.getenv(
        "NVIDIA_TEXT_MODELS",
        "nvidia/nemotron-3.5-lightning-30b-a3b,openai/gpt-oss-120b",
    )
    nvidia_text_models = [(m.strip(), 30) for m in _nvidia_models_raw.split(",") if m.strip()]
    # To add more NVIDIA text models, set NVIDIA_TEXT_MODELS in .env. The startup
    # capability gate auto-drops any that 404 or can't do structured output, so
    # only proven-agentic models ever reach the chain (see STRATEGY.md).
    for key_idx, key in enumerate(nvidia_keys):
        for model_name, timeout_s in nvidia_text_models:
            clients.append(
                ModelClient(
                    name=f"nvidia-text:{model_name}:{key_idx}",
                    client=ChatOpenAI(
                        model=model_name,
                        api_key=key,
                        base_url=base,
                        temperature=0.0,
                        timeout=timeout_s,
                        max_tokens=text_max_tokens,
                        default_headers=headers,
                    ),
                    provider="nvidia",
                    pipeline="text",
                    credential_id=_credential_fingerprint(key),
                )
            )
    if nvidia_keys:
        logger.info(
            "TEXT ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_keys),
            len(nvidia_text_models),
            len(nvidia_keys) * len(nvidia_text_models),
        )

    # ── Cloudflare Workers AI — OpenAI-compatible free allocation ──
    # Requires both an account id and an API token. Multiple models share the
    # same account-level neuron allocation, so model diversity is useful for
    # capability/latency failover but does not multiply quota.
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_tokens = _collect_keys(
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_TOKENS"
    )
    if _env_flag("CLOUDFLARE_ENABLED", True) and cloudflare_tokens:
        if not cloudflare_account_id:
            logger.warning(
                "TEXT ── Cloudflare token configured without CLOUDFLARE_ACCOUNT_ID; skipping"
            )
        else:
            cloudflare_base = os.getenv("CLOUDFLARE_BASE_URL", "").strip() or (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{cloudflare_account_id}/ai/v1"
            )
            cloudflare_models = [
                m.strip()
                for m in os.getenv(
                    "CLOUDFLARE_TEXT_MODELS",
                    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                ).split(",")
                if m.strip()
            ]
            for key_idx, token in enumerate(cloudflare_tokens):
                for model_name in cloudflare_models:
                    clients.append(
                        ModelClient(
                            name=f"cloudflare:{model_name}:{key_idx}",
                            client=ChatOpenAI(
                                model=model_name,
                                api_key=token,
                                base_url=cloudflare_base,
                                temperature=0.0,
                                timeout=30,
                                max_tokens=min(
                                    _int_env("CLOUDFLARE_MAX_TOKENS", 2048, 256),
                                    text_max_tokens,
                                ),
                                default_headers=headers,
                            ),
                            provider="cloudflare",
                            pipeline="text",
                            credential_id=_credential_fingerprint(cloudflare_account_id),
                        )
                    )
            logger.info(
                "TEXT ── Cloudflare Workers AI: %d token(s) × %d models = %d instances",
                len(cloudflare_tokens),
                len(cloudflare_models),
                len(cloudflare_tokens) * len(cloudflare_models),
            )

    # ── Google Gemini — primary worker (role ordering is applied below) ──
    gemini_text_keys = _collect_keys("GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACKS")
    if gemini_text_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_text_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite")
            for idx, key in enumerate(gemini_text_keys):
                clients.append(
                    ModelClient(
                        name=f"gemini-text:{gemini_text_model}:{idx}",
                        client=ChatGoogleGenerativeAI(
                            model=gemini_text_model,
                            google_api_key=key,
                            temperature=0.0,
                            timeout=30,
                            max_output_tokens=text_max_tokens,
                        ),
                        provider="google",
                        pipeline="text",
                        credential_id=_credential_fingerprint(key),
                    )
                )
            logger.info(
                "TEXT ── Google Gemini: %d keys loaded (%s)",
                len(gemini_text_keys),
                gemini_text_model,
            )
        except ImportError:
            logger.warning("TEXT ── langchain-google-genai not installed, skipping Gemini text fallback")
        except Exception as e:
            logger.error("TEXT ── Gemini text fallback bootstrap failed: %s", e)

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION Pipeline — Google Gemini (proven) + NVIDIA NIM Vision (fallback)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_vision_pipeline() -> list[ModelClient]:
    """VISION-ONLY pipeline for Executor. ALWAYS receives images."""
    clients: list[ModelClient] = []

    # ── Google Gemini — PRIMARY VISION (2 keys, gemma-4 vision model) ──
    gemini_keys = _collect_keys(
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACKS",
        "VISION_GOOGLE_API_KEY",
        "VISION_GOOGLE_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY",
    )
    if gemini_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("WORKER_VLM_MODEL", "gemma-4-31b-it")
            for idx, key in enumerate(gemini_keys):
                clients.append(
                    ModelClient(
                        name=f"gemini-vision:{model}:{idx}",
                        client=ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=key,
                            temperature=0.0,
                            timeout=45,
                        ),
                        provider="google",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(key),
                    )
                )
            logger.info("VISION ── Google Gemini (PRIMARY): %d keys loaded (%s)", len(gemini_keys), model)
        except ImportError:
            logger.warning("VISION ── langchain-google-genai not installed, skipping Gemini")
        except Exception as e:
            logger.error("VISION ── Gemini bootstrap failed: %s", e)

    # ── NVIDIA NIM Vision — SECONDARY (multi-model × ALL NVIDIA keys) ──
    # Collect all NVIDIA keys so every vision model gets tried on every key.
    nvidia_vision_keys = _collect_keys(
        "NVIDIA_VISION_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_API_KEYS",
    )
    if nvidia_vision_keys:
        from langchain_openai import ChatOpenAI

        base = os.getenv("NVIDIA_VISION_BASE_URL", "https://integrate.api.nvidia.com/v1")

        # OPTIONAL + env-configurable via NVIDIA_VISION_MODELS (comma-separated),
        # mirroring NVIDIA_TEXT_MODELS. Default leads with Llama 4 Maverick (the
        # chosen vision model); the rest are optional fallbacks. All probed
        # 2026-06 to respond <1s; the startup capability gate drops any that 404
        # or can't structure. To use ONLY Maverick, set
        # NVIDIA_VISION_MODELS="meta/llama-4-maverick-17b-128e-instruct".
        _nv_vision_raw = os.getenv(
            "NVIDIA_VISION_MODELS",
            "meta/llama-4-maverick-17b-128e-instruct,"  # Llama 4 Maverick — primary
            "meta/llama-3.2-11b-vision-instruct,"  # fast, lightweight fallback
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1,"  # fast nano VL fallback
            "meta/llama-3.2-90b-vision-instruct",  # strong 90B fallback
        )
        _nv_vision_timeout = int(os.getenv("NVIDIA_VISION_TIMEOUT", "30"))
        nvidia_vision_models = [
            (m.strip(), _nv_vision_timeout) for m in _nv_vision_raw.split(",") if m.strip()
        ]
        for key_idx, key in enumerate(nvidia_vision_keys):
            for model_name, timeout_s in nvidia_vision_models:
                clients.append(
                    ModelClient(
                        name=f"nvidia-vision:{model_name}:{key_idx}",
                        client=ChatOpenAI(
                            model=model_name,
                            api_key=key,
                            base_url=base,
                            temperature=0.0,
                            timeout=timeout_s,
                        ),
                        provider="nvidia",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(key),
                    )
                )
        logger.info(
            "VISION ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_vision_keys),
            len(nvidia_vision_models),
            len(nvidia_vision_keys) * len(nvidia_vision_models),
        )

    # ── Cloudflare Workers AI vision — optional, account-level free allocation ──
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_tokens = _collect_keys(
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_TOKENS"
    )
    if (_env_flag("CLOUDFLARE_ENABLED", True)
            and _env_flag("CLOUDFLARE_VISION_ENABLED", True)
            and cloudflare_account_id and cloudflare_tokens):
        cloudflare_vision_models = [
            m.strip()
            for m in os.getenv(
                "CLOUDFLARE_VISION_MODELS",
                "@cf/meta/llama-3.2-11b-vision-instruct",
            ).split(",")
            if m.strip()
        ]
        for key_idx, token in enumerate(cloudflare_tokens):
            for model_name in cloudflare_vision_models:
                clients.append(
                    ModelClient(
                        name=f"cloudflare-vision:{model_name}:{key_idx}",
                        client=CloudflareNativeVisionClient(
                            account_id=cloudflare_account_id,
                            api_token=token,
                            model=model_name,
                            timeout=45,
                        ),
                        provider="cloudflare",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(cloudflare_account_id),
                    )
                )
        logger.info(
            "VISION ── Cloudflare Workers AI: %d token(s) × %d models = %d instances",
            len(cloudflare_tokens),
            len(cloudflare_vision_models),
            len(cloudflare_tokens) * len(cloudflare_vision_models),
        )

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO Pipeline — bounded calls for detected survey media questions
# ═══════════════════════════════════════════════════════════════════════════════


def _build_audio_pipeline() -> list[ModelClient]:
    """AUDIO-ONLY Gemini chain, invoked solely for detected media questions."""
    if not _env_flag("SURVEY_AUDIO_ENABLED", True):
        return []
    keys = _collect_keys(
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEY_FALLBACKS",
        "VISION_GOOGLE_API_KEY",
        "VISION_GOOGLE_API_KEY_FALLBACKS",
    )
    if not keys:
        return []
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv(
            "SURVEY_AUDIO_MODEL",
            os.getenv("WORKER_VLM_MODEL", "gemini-3.5-flash"),
        ).strip()
        clients = [
            ModelClient(
                name=f"gemini-audio:{model}:{idx}",
                client=ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=key,
                    temperature=0.0,
                    timeout=35,
                ),
                provider="google",
                pipeline="audio",
                credential_id=_credential_fingerprint(key),
            )
            for idx, key in enumerate(keys)
        ]
        logger.info("AUDIO ── Google Gemini: %d keys loaded (%s)", len(keys), model)
        return clients
    except Exception as exc:  # noqa: BLE001 - absence falls back to a constrained guess
        logger.warning("AUDIO ── Gemini bootstrap unavailable: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  PREMIUM Pipeline — one paid key/model serves BOTH text and vision (current)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_premium_pipeline() -> tuple[list["ModelClient"], list["ModelClient"]]:
    """Premium mode: a single top-tier multimodal model drives everything.

    Trusted by definition — no probe, no capability gate, no free-tier fallback
    juggling. OpenAI-compatible by default (covers OpenAI / OpenRouter / Together
    / Fireworks / any gateway); PREMIUM_PROVIDER=google uses the Gemini client.
    Returns (text_clients, vision_clients); audio remains a separate pipeline.
    """
    cfg = _premium_config()
    keys, model = cfg["keys"], cfg["model"]
    if not keys or not model:
        logger.error("PREMIUM mode needs PREMIUM_API_KEY and PREMIUM_MODEL — none/partial set.")
        return [], []

    def _mk(m: str, key: str, idx: int, pipeline: str) -> "ModelClient":
        if cfg["provider"] == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                model=m, google_api_key=key, temperature=0.0, timeout=cfg["timeout"]
            )
        else:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(
                model=m, api_key=key, base_url=cfg["base_url"], temperature=0.0, timeout=cfg["timeout"]
            )
        return ModelClient(
            name=f"premium-{pipeline}:{m}:{idx}", client=client,
            provider="premium", pipeline=pipeline,
            credential_id=_credential_fingerprint(key),
        )

    text = [_mk(model, k, i, "text") for i, k in enumerate(keys)]
    vmodel = cfg["vision_model"] or model
    vision = [_mk(vmodel, k, i, "vision") for i, k in enumerate(keys)]
    logger.info(
        "⭐ PREMIUM mode: text='%s', vision='%s' on %d key(s) — no probe, no gate", model, vmodel, len(keys)
    )
    return text, vision


def _build_premium_audio_pipeline() -> list["ModelClient"]:
    """Use the premium Google key for audio without changing the legacy tuple API."""
    cfg = _premium_config()
    if (cfg["provider"] != "google" or not cfg["keys"]
            or not _env_flag("SURVEY_AUDIO_ENABLED", True)):
        return []
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv(
        "SURVEY_AUDIO_MODEL",
        cfg["vision_model"] or cfg["model"],
    ).strip()
    return [
        ModelClient(
            name=f"premium-audio:{model}:{idx}",
            client=ChatGoogleGenerativeAI(
                model=model,
                google_api_key=key,
                temperature=0.0,
                timeout=cfg["timeout"],
            ),
            provider="premium",
            pipeline="audio",
            credential_id=_credential_fingerprint(key),
        )
        for idx, key in enumerate(cfg["keys"])
    ]
