"""External knowledge source retriever — GitHub Issues, Stack Overflow, Web Search.

Each source is independently queried with its own error handling.
A single source failure does not affect other sources or the main pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from enterprise_agentic_rag.config.settings import get_settings

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

_STACKOVERFLOW_API = "https://api.stackexchange.com/2.3"
_GITHUB_API = "https://api.github.com"


class ExternalRetriever:
    """Retriever for external knowledge sources.

    Supports:
    - GitHub Issues search
    - Stack Overflow / Stack Exchange search
    - Web search (via configurable provider)

    All sources are optional and can be configured via environment variables.
    """

    def __init__(self) -> None:
        self._settings = get_settings().external_search
        self._http_client = None

    @property
    def available(self) -> bool:
        """Check if external search is enabled and at least one source is configured."""
        if not self._settings.enabled:
            return False
        return bool(
            (self._settings.github_repos and self._settings.github_token)
            or self._settings.stackexchange_key
            or (self._settings.web_search_provider != "none" and self._settings.web_search_api_key)
        )

    async def _get_client(self):
        """Lazy HTTP client creation."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                timeout=self._settings.external_timeout_seconds,
                headers={"User-Agent": "Enterprise-Agentic-RAG/1.0"},
            )
        return self._http_client

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search external sources for relevant content.

        Args:
            query: Search query string.
            sources: Which sources to query (default: all configured).
            top_k: Max results per source.

        Returns:
            List of unified result dicts with keys:
            - source_type: "github" | "stackoverflow" | "web"
            - title: Human-readable title
            - content: Text content / snippet
            - url: Link to the original source
            - score: Relevance score (0.0–1.0)
        """
        if not self.available:
            return []

        if sources is None:
            sources = self._default_sources()

        tasks = []
        for source in sources:
            if source == "github" and self._settings.github_token and self._settings.github_repos:
                tasks.append(self._search_github_issues(query, top_k))
            elif source == "stackoverflow" and self._settings.stackexchange_key:
                tasks.append(self._search_stackoverflow(query, top_k))
            elif source == "web" and self._settings.web_search_provider != "none":
                tasks.append(self._search_web(query, top_k))

        if not tasks:
            return []

        # Run all sources in parallel with per-source error isolation
        results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[dict[str, Any]] = []
        for result in results_per_source:
            if isinstance(result, Exception):
                logger.warning("External search source failed: %s", result)
                continue
            if isinstance(result, list):
                all_results.extend(result)

        return all_results[:top_k * len(sources)]

    def _default_sources(self) -> list[str]:
        """Return the list of configured sources."""
        sources: list[str] = []
        if self._settings.github_token and self._settings.github_repos:
            sources.append("github")
        if self._settings.stackexchange_key:
            sources.append("stackoverflow")
        if self._settings.web_search_provider != "none" and self._settings.web_search_api_key:
            sources.append("web")
        return sources

    # ===========================================================================
    # GitHub Issues search
    # ===========================================================================

    async def _search_github_issues(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search GitHub Issues across configured repos."""
        results: list[dict[str, Any]] = []
        client = await self._get_client()

        # Search across all configured repos
        repo_queries = " OR ".join(f"repo:{r}" for r in self._settings.github_repos)
        search_query = f"{query} {repo_queries} is:issue"

        try:
            resp = await client.get(
                f"{_GITHUB_API}/search/issues",
                params={
                    "q": search_query,
                    "per_page": top_k,
                    "sort": "relevance",
                },
                headers={
                    "Authorization": f"Bearer {self._settings.github_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("items", [])[:top_k]:
                    results.append({
                        "source_type": "github",
                        "title": item.get("title", ""),
                        "content": (item.get("body") or "")[:500],
                        "url": item.get("html_url", ""),
                        "score": round(0.5 + 0.3 * float(item.get("score", 1) / 10), 2),
                        "source": f"github:{item.get('repository_url', '').split('/repos/')[-1]}",
                    })
            elif resp.status_code == 403:
                logger.warning("GitHub API rate limited")
            else:
                logger.debug("GitHub API returned %s", resp.status_code)
        except Exception as exc:
            logger.warning("GitHub Issues search failed: %s", exc)

        return results

    # ===========================================================================
    # Stack Overflow search
    # ===========================================================================

    async def _search_stackoverflow(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search Stack Overflow / Stack Exchange for relevant Q&A."""
        results: list[dict[str, Any]] = []
        client = await self._get_client()

        try:
            resp = await client.get(
                f"{_STACKOVERFLOW_API}/search/advanced",
                params={
                    "q": query,
                    "pagesize": top_k,
                    "order": "desc",
                    "sort": "relevance",
                    "site": "stackoverflow",
                    "key": self._settings.stackexchange_key,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("items", [])[:top_k]:
                    results.append({
                        "source_type": "stackoverflow",
                        "title": item.get("title", ""),
                        "content": (item.get("body_markdown") or "")[:500],
                        "url": item.get("link", ""),
                        "score": round(float(item.get("score", 1) / 100), 2),
                        "source": "stackoverflow",
                    })
            else:
                logger.debug("StackExchange API returned %s", resp.status_code)
        except Exception as exc:
            logger.warning("Stack Overflow search failed: %s", exc)

        return results

    # ===========================================================================
    # Web search (configurable provider)
    # ===========================================================================

    async def _search_web(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search the web using the configured provider (SerpAPI, Bing, or mock).

        Currently implements a mock/placeholder that returns structured empty results.
        Providers can be added via the adapters pattern (like tools/adapters/).
        """
        provider = self._settings.web_search_provider
        api_key = self._settings.web_search_api_key

        if provider == "serpapi":
            return await self._search_serpapi(query, top_k, api_key)
        elif provider == "bing":
            return await self._search_bing(query, top_k, api_key)
        else:
            # No provider configured — return empty
            logger.debug("No web search provider configured (provider=%s)", provider)
            return []

    async def _search_serpapi(
        self, query: str, top_k: int, api_key: str,
    ) -> list[dict[str, Any]]:
        """Search using SerpAPI (Google Search)."""
        results: list[dict[str, Any]] = []
        client = await self._get_client()

        try:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": api_key,
                    "num": top_k,
                    "engine": "google",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("organic_results", [])[:top_k]:
                    results.append({
                        "source_type": "web",
                        "title": item.get("title", ""),
                        "content": item.get("snippet", "")[:500],
                        "url": item.get("link", ""),
                        "score": 0.4,
                        "source": "web:serpapi",
                    })
        except Exception as exc:
            logger.warning("SerpAPI search failed: %s", exc)

        return results

    async def _search_bing(
        self, query: str, top_k: int, api_key: str,
    ) -> list[dict[str, Any]]:
        """Search using Bing Web Search API."""
        results: list[dict[str, Any]] = []
        client = await self._get_client()

        try:
            resp = await client.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params={"q": query, "count": top_k},
                headers={"Ocp-Apim-Subscription-Key": api_key},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("webPages", {}).get("value", [])[:top_k]:
                    results.append({
                        "source_type": "web",
                        "title": item.get("name", ""),
                        "content": item.get("snippet", "")[:500],
                        "url": item.get("url", ""),
                        "score": 0.4,
                        "source": "web:bing",
                    })
        except Exception as exc:
            logger.warning("Bing search failed: %s", exc)

        return results

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
