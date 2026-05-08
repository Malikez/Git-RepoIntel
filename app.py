import os
import re
import json
import logging
import time
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional

from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GitLeads")

app = FastAPI(title="GitLeads Intent Intelligence API", version="4.0")
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

try:
    ai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    VERTEX_AVAILABLE = True
    logger.info("Vertex AI Client initialized.")
except Exception as e:
    logger.error(f"Vertex AI init error: {e}")
    VERTEX_AVAILABLE = False

# ==========================================
# 2. KEYWORD TAXONOMY
# ==========================================

# Grouped by business intent — used for smarter signal categorization
INTENT_TAXONOMY = {
    "scaling_infra": [
        "kubernetes", "k8s", "helm", "eks", "gke", "aks", "docker",
        "containerd", "kustomize", "istio", "service-mesh", "envoy", "linkerd",
    ],
    "cloud_migration": [
        "migration", "migrate", "aws", "gcp", "azure", "terraform", "pulumi",
        "cdk", "cloudformation", "multi-cloud", "hybrid-cloud",
    ],
    "security_posture": [
        "auth", "oauth", "oidc", "sso", "rbac", "iam", "vault", "secrets",
        "tls", "encryption", "compliance", "soc2", "audit", "snyk", "trivy", "dependabot",
    ],
    "observability": [
        "prometheus", "grafana", "datadog", "otel", "opentelemetry", "tracing",
        "metrics", "logging", "alerting", "pagerduty", "loki", "tempo", "jaeger",
    ],
    "ci_cd": [
        "ci", "cd", "pipeline", "github-actions", "jenkins", "circleci",
        "gitlab", "argocd", "flux", "gitops", "deployment", "release",
    ],
    "data_platform": [
        "kafka", "spark", "flink", "dbt", "snowflake", "bigquery", "databricks",
        "airflow", "etl", "data-pipeline", "warehouse",
    ],
    "platform_eng": [
        "platform", "developer-experience", "devex", "internal-tools",
        "backstage", "idp", "golden-path", "service-catalog",
    ],
    "ai_ml": [
        "llm", "ml", "ai", "model", "inference", "training", "embeddings",
        "vector", "langchain", "rag", "openai", "anthropic", "huggingface", "gpu",
    ],
}

ALL_DEFAULT_KEYWORDS = [kw for kws in INTENT_TAXONOMY.values() for kw in kws]

NOISE_COMMITS = {
    "merge pull request", "update readme", "typo", "chore", "version packages",
    "bump", "lint", "cleanup", "release", ".gitignore", "dependabot", "merge branch",
    "update dependencies", "fix typo", "minor fix", "whitespace",
}

# Cache
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes

# ── FIX 1: Semaphore ──────────────────────────────────────────────────────────
# Root cause of Shopify's 93-second run: 15 repos × 5 endpoints = 75 requests
# fired simultaneously, overwhelming GitHub's connection pool and causing mass
# silent disconnects. This caps concurrency to 12 at any given moment.
GITHUB_SEMAPHORE = asyncio.Semaphore(12)


# ==========================================
# 3. DATA MODELS
# ==========================================

class AnalyzeRequest(BaseModel):
    github_org: str
    custom_keywords: List[str] = Field(default_factory=list)
    business_context: str = "General B2B SaaS"
    industry: str = "General"
    seller_product: str = ""  # What the GitLeads customer sells — personalises AI output


class AgentInsight(BaseModel):
    signal: str = Field(
        description="Short signal name, e.g. 'Active Cloud Migration' or 'Security Hardening Initiative'"
    )
    summary: str = Field(
        description="One-sentence plain-English summary of what this organization is doing right now."
    )
    evidence: List[str] = Field(
        description=(
            "3-6 specific evidence items. Each must cite a real repo name, PR title, "
            "issue label, or commit message from the payload. No invented content."
        )
    )
    business_insight: str = Field(
        description="Why this activity matters commercially. What unmet need does it reveal?"
    )
    opportunity: str = Field(
        description=(
            "Specific, actionable sales angle tied to the seller's product. "
            "If no fit exists, say so clearly."
        )
    )
    urgency: str = Field(
        description="High, Medium, or Low — followed by one sentence of reasoning."
    )
    confidence: int = Field(
        description=(
            "0-100. How well-evidenced this analysis is. "
            "80-100: multi-source convergence. 50-79: moderate, 2 sources. "
            "25-49: weak, 1 source. <25: insufficient data."
        )
    )
    activity_level: str = Field(description="High, Medium, or Low.")
    tech_stack_signals: List[str] = Field(
        description="Technologies or platforms actively in use or being adopted, from evidence only."
    )
    intent_category: str = Field(
        description=(
            "Primary intent bucket: scaling_infra | cloud_migration | security_posture | "
            "observability | ci_cd | data_platform | platform_eng | ai_ml | "
            "product_development | mixed"
        )
    )
    is_product_builder: bool = Field(
        description=(
            "True if this org BUILDS the technology (e.g. HashiCorp builds Terraform). "
            "False if they are a user or adopter of external technologies."
        )
    )
    # ── FIX 3: recommended_action ─────────────────────────────────────────────
    # Gives the frontend a machine-readable verdict so CRM workflows can be
    # triggered without parsing free text from the opportunity field.
    # Four possible values: ENGAGE_NOW | MONITOR | PARTNER_EXPLORE | SKIP
    recommended_action: str = Field(
        description=(
            "ENGAGE_NOW: Strong signal match — reach out this week. "
            "MONITOR: Signals emerging but not mature — check again in 30 days. "
            "PARTNER_EXPLORE: Potential integration or BD partner, not a direct sale. "
            "SKIP: No fit, direct competitor, or insufficient signal to justify outreach."
        )
    )


# ==========================================
# 4. GITHUB FETCHING
# ==========================================

def extract_org_name(github_input: str) -> str:
    """Handles full GitHub URLs or plain org names."""
    match = re.search(r"github\.com/([^/?]+)", github_input)
    if match:
        return match.group(1).strip()
    return github_input.strip().strip("/")


# ── FIX 1 (continued): Replaced fetch function ───────────────────────────────
# Changes from old version:
#   1. Wrapped in GITHUB_SEMAPHORE — caps concurrency, prevents connection storms
#   2. asyncio.wait_for(..., timeout=8.0) — hard per-request timeout; one slow
#      repo can no longer hold up the entire asyncio.gather batch
#   3. Staggered retry delays — avoids all timed-out requests retrying at once
#   4. Logs exception type explicitly — no more empty "HTTP error for ...: " lines
async def fetch_github_data_async(url: str, client: httpx.AsyncClient) -> Any:
    """Fetch with caching, semaphore-limited concurrency, and hard timeout."""

    # Cache check
    if url in CACHE and time.time() - CACHE[url]["timestamp"] < CACHE_TTL:
        return CACHE[url]["data"]

    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    async with GITHUB_SEMAPHORE:  # Never more than 12 concurrent requests
        for attempt in range(3):
            try:
                # Hard 8-second timeout — no single request can block the pipeline
                response = await asyncio.wait_for(
                    client.get(url, headers=headers),
                    timeout=8.0,
                )

                if response.status_code in (403, 429):
                    reset_time = int(
                        response.headers.get("X-RateLimit-Reset", time.time() + 30)
                    )
                    sleep_time = max(min(reset_time - time.time(), 15), 2)
                    logger.warning(
                        f"Rate limited ({response.status_code}) on {url}. "
                        f"Sleeping {sleep_time:.0f}s."
                    )
                    await asyncio.sleep(sleep_time)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    CACHE[url] = {"data": data, "timestamp": time.time()}
                    return data

                logger.warning(f"Status {response.status_code} for {url}")
                return []

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                # Staggered pause — avoids synchronized retry storms
                await asyncio.sleep(1.5 * (attempt + 1))
                continue

            except httpx.RequestError as e:
                # Log exception type explicitly — no more empty error messages
                logger.error(
                    f"[{type(e).__name__}] Request error on attempt {attempt + 1} "
                    f"for {url}: {e}"
                )
                await asyncio.sleep(2 ** attempt)
                continue

    logger.warning(f"All 3 attempts exhausted for {url}. Returning empty.")
    return []


# ==========================================
# 5. SIGNAL EXTRACTION (ENRICHED)
# ==========================================

def safe_str(val: Any, limit: int = 120) -> str:
    return str(val)[:limit] if val else ""


def extract_pr_signals(prs: list) -> List[str]:
    """PR titles are far more informative than commit messages."""
    signals = []
    noise = {"dependencies", "chore", "bump", "typo", "readme", "lint", "dependabot"}
    for pr in prs[:20]:
        title = safe_str(pr.get("title", "")).lower()
        if any(n in title for n in noise):
            continue
        if len(title) > 15:
            signals.append(pr.get("title", ""))
    return signals[:12]


def extract_issue_signals(issues: list) -> Tuple[List[str], List[str]]:
    """Issues with labels reveal engineering priorities directly."""
    titles = []
    labels_found = []
    high_value_labels = {
        "migration", "security", "performance", "scalability", "infrastructure",
        "cloud", "kubernetes", "k8s", "auth", "observability", "ci/cd",
        "breaking-change", "epic", "milestone", "feature", "platform",
        "architecture", "refactor",
    }
    for issue in issues[:20]:
        title = safe_str(issue.get("title", ""))
        if len(title) > 15:
            titles.append(title)
        for label in issue.get("labels", []):
            lname = label.get("name", "").lower()
            if any(hv in lname for hv in high_value_labels):
                labels_found.append(lname)

    return titles[:8], list(set(labels_found))[:10]


def extract_repo_metadata(repos: list) -> Tuple[List[Dict], List[str], Dict[str, int]]:
    """Extract topics, languages, and structured repo metadata."""
    enriched_repos = []
    all_topics = []
    language_counts: Dict[str, int] = {}

    for repo in repos:
        topics = repo.get("topics", [])
        lang = repo.get("language") or "Unknown"
        all_topics.extend(topics)

        if lang != "Unknown":
            language_counts[lang] = language_counts.get(lang, 0) + 1

        enriched_repos.append({
            "name": repo.get("name", ""),
            "description": safe_str(repo.get("description", ""), 200),
            "topics": topics,
            "language": lang,
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "open_issues": repo.get("open_issues_count", 0),
            "created_at": repo.get("created_at", "")[:10],
            "pushed_at": repo.get("pushed_at", "")[:10],
            "is_fork": repo.get("fork", False),
        })

    return enriched_repos, list(set(all_topics)), language_counts


def extract_workflow_tools(workflows: list) -> List[str]:
    """Parse .github/workflows to detect CI/CD tools and integrations."""
    tools_found = set()
    tool_patterns = {
        "GitHub Actions": "uses:",
        "AWS": "aws-actions",
        "GCP": "google-github-actions",
        "Azure": "azure/",
        "Terraform": "hashicorp/terraform",
        "Docker": "docker/",
        "Kubernetes": "kubectl",
        "ArgoCD": "argocd",
        "Snyk": "snyk",
        "Datadog": "datadog",
        "Slack Notifications": "slack",
    }
    for workflow in workflows[:5]:
        content = workflow.get("content_decoded", "")
        for tool, pattern in tool_patterns.items():
            if pattern.lower() in content.lower():
                tools_found.add(tool)
    return list(tools_found)


# ==========================================
# 6. SIGNAL SCORING (RECALIBRATED)
# ==========================================

def score_signals(
    user_kws: list,
    enriched_repos: List[Dict],
    commits: list,
    pr_titles: List[str],
    issue_titles: List[str],
    issue_labels: List[str],
    topics: List[str],
) -> Tuple[Dict[str, List[str]], int]:
    """
    ── FIX 2: Recalibrated scoring — 100 is now genuinely rare ──────────────

    Score tiers:
      80-100 : Multiple sources converging on the same intent (topics + PRs +
               issue labels all pointing the same way). Only strong, active orgs
               with rich public signals should reach this range.
      50-79  : Clear signal in 2+ sources. Keyword hits in topics + PRs, or
               strong user keyword matches with supporting commits.
      25-49  : Weak or single-source. A few keyword matches but scattered,
               no cross-source convergence.
      0-24   : Very sparse. Quiet org, product builder self-reference, or no
               seller product fit at all.

    Four independent dimensions — each caps low so no single one inflates total:
      1. User keyword score    (max 30) — seller-defined keywords, highest priority
      2. Taxonomy breadth      (max 35) — how many intent categories have hits
      3. Source diversity      (max 20) — signals found across topics/PRs/issues
      4. Activity volume       (max 15) — raw data richness bonus

    Old weights that caused every org to score 100:
      user_score = len(hits) * 20        → hits ceiling at just 5 matches
      taxonomy   = min(len(v) * 5, 25)  → one noisy category was worth 25 pts
    """

    cleaned_user_kws = [k.lower().strip() for k in user_kws if len(k.strip()) >= 2]

    category_hits: Dict[str, List[str]] = {cat: [] for cat in INTENT_TAXONOMY}
    category_hits["user_custom"] = []

    # Track which source types each user keyword was found in (for convergence bonus)
    user_kw_sources: Dict[str, set] = {}

    def match_text(text: str, keywords: list) -> List[str]:
        text = text.lower()
        return [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text)]

    # Bucket all signal sources by type — enables cross-source convergence scoring
    source_buckets: Dict[str, List[str]] = {
        "topics": [" ".join(r.get("topics", [])) for r in enriched_repos] + topics,
        "pr":     pr_titles,
        "issue":  issue_titles + issue_labels,
        "repo":   [r["name"] + " " + r["description"] for r in enriched_repos],
        "commit": [
            c.get("commit", {}).get("message", "").split("\n")[0]
            for c in commits[:30]
        ],
    }

    for source_type, texts in source_buckets.items():
        for text in texts:
            if not text:
                continue

            # User custom keywords
            for kw in match_text(text, cleaned_user_kws):
                if kw not in category_hits["user_custom"]:
                    category_hits["user_custom"].append(kw)
                if kw not in user_kw_sources:
                    user_kw_sources[kw] = set()
                user_kw_sources[kw].add(source_type)

            # Taxonomy keywords
            for cat, kws in INTENT_TAXONOMY.items():
                for kw in match_text(text, kws):
                    if kw not in category_hits[cat]:
                        category_hits[cat].append(kw)

    # ── Dimension 1: User keyword score (max 30) ──────────────────────────────
    # 5 pts per unique keyword hit (was 20 — was hitting ceiling at 5 matches)
    # Convergence bonus: +3 pts if same keyword found in 3+ different source types
    user_base = min(len(category_hits["user_custom"]) * 5, 20)
    convergence_bonus = sum(
        3 for kw, sources in user_kw_sources.items() if len(sources) >= 3
    )
    user_score = min(user_base + convergence_bonus, 30)

    # ── Dimension 2: Taxonomy breadth score (max 35) ──────────────────────────
    # Each category caps at 7 pts (was 25 — one noisy category was worth 25 pts)
    # Rewards orgs active across multiple intent categories
    active_categories = {
        k: v for k, v in category_hits.items() if v and k != "user_custom"
    }
    taxonomy_score = min(
        sum(min(len(v) * 2, 7) for v in active_categories.values()),
        35,
    )

    # ── Dimension 3: Source diversity score (max 20) ──────────────────────────
    # Rewards finding signals across different source types, not just commits.
    # "We found keywords in topics AND PR titles AND issue labels" = real signal.
    sources_with_hits = sum([
        5 if any(
            match_text(t, cleaned_user_kws + ALL_DEFAULT_KEYWORDS)
            for t in source_buckets["topics"] if t
        ) else 0,
        5 if any(
            match_text(t, cleaned_user_kws + ALL_DEFAULT_KEYWORDS)
            for t in source_buckets["pr"] if t
        ) else 0,
        5 if any(
            match_text(t, cleaned_user_kws + ALL_DEFAULT_KEYWORDS)
            for t in source_buckets["issue"] if t
        ) else 0,
        5 if issue_labels else 0,  # Having labelled issues = org has mature triage
    ])
    source_diversity_score = min(sources_with_hits, 20)

    # ── Dimension 4: Activity volume score (max 15) ───────────────────────────
    # Small bonus for raw data richness — more data = more reliable score
    volume_score = min(
        (1 if len(pr_titles) >= 5 else 0) * 5
        + (1 if len(issue_titles) >= 5 else 0) * 5
        + (1 if len(topics) >= 10 else 0) * 5,
        15,
    )

    total = user_score + taxonomy_score + source_diversity_score + volume_score

    logger.info(
        f"Score breakdown — user:{user_score} taxonomy:{taxonomy_score} "
        f"diversity:{source_diversity_score} volume:{volume_score} total:{total}"
    )

    return category_hits, int(total)


# ==========================================
# 7. AI AGENT
# ==========================================

def call_intent_agent(payload: dict) -> dict:
    if not VERTEX_AVAILABLE:
        return _fallback_insight()

    system_instruction = """
You are the core intelligence engine of GitLeads — a B2B sales intelligence platform that converts public GitHub activity into buying-signal insights for DevTools, Cloud Infrastructure, Security, and SaaS companies.

Your role: Analyze structured GitHub evidence and produce concise, accurate, commercially-relevant intelligence — the kind that helps a sales rep decide whether to reach out and exactly what to say.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. USE ONLY the evidence in the payload. No hallucination. No invented repos, tools, or timelines.
2. DISAMBIGUATE PRODUCT BUILDERS vs. TECHNOLOGY ADOPTERS:
   - If the org BUILDS the technology (e.g. HashiCorp builds Terraform, Grafana builds Grafana), do NOT treat their own product activity as external adoption intent.
   - For product builders, analyse ONLY their INTERNAL tooling signals: how they run CI/CD, what they use for observability, their security stack, their cloud infrastructure choices.
   - Mark `is_product_builder: true` for these orgs.
3. NEVER confuse repo maintenance (bug fixes, docs updates, version bumps) with intent signals.
4. Prioritise signals in this order: PR titles > issue titles/labels > repo topics > repo descriptions > commit messages.
5. Cite specific repo names, PR titles, or issue labels. Never write vague generalisations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL INTELLIGENCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Map observed activity to these business intents:

| Activity Signal                              | Business Intent                               |
|----------------------------------------------|-----------------------------------------------|
| New repos named platform/core/infra/dx       | Internal platform investment or rebuild        |
| K8s/Helm/service-mesh PRs or topics          | Container orchestration adoption/maturity      |
| Auth/OIDC/RBAC/vault activity                | Security hardening or compliance initiative    |
| Terraform/Pulumi/CDK activity (non-builders) | Cloud infrastructure expansion                 |
| OTEL/Prometheus/Grafana/Datadog topics       | Observability stack investment                 |
| LLM/embeddings/inference repos               | AI product development or AI infra adoption    |
| Data pipeline, dbt, Kafka, warehouse topics  | Data platform modernisation                    |
| Large contributor count + new repos          | Team growth / active hiring phase              |
| GitHub Actions + ArgoCD/Flux                 | GitOps maturity or DevOps transformation       |
| Security labels, Snyk, trivy, audit topics   | AppSec / supply chain security investment      |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELLER ALIGNMENT RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The payload includes `seller_product`. Use it to make `opportunity` highly specific:
- If the seller sells observability tooling and the org is adopting k8s → "They are adding container orchestration without native observability — direct outreach angle."
- If the seller sells cost optimisation and the org shows multi-cloud sprawl → "Multi-cloud repo pattern suggests unmanaged cloud spend."
- If signals don't match the seller's product at all, say so honestly and set recommended_action to SKIP.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

80-100 : Strong multi-source evidence (topics + PRs + issues + commits all align).
50-79  : Moderate — 2 sources, clear intent direction.
25-49  : Weak — 1 source or ambiguous signals.
<25    : Insufficient data — say so explicitly.

NEVER inflate confidence to sound authoritative. A 30 with honest reasoning is more valuable than a fabricated 80.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED ACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Set `recommended_action` using this decision logic. It must be EXACTLY one of four values:

ENGAGE_NOW      → confidence >= 60 AND signals clearly match seller_product
                  AND is_product_builder is False
                  OR is_product_builder is True but their INTERNAL tooling signals
                  match seller_product strongly

MONITOR         → confidence 35-59, OR signals are emerging but not yet mature
                  (e.g. early-stage repos, few PRs, low issue label activity)
                  OR is_product_builder is True with only weak internal tooling signals

PARTNER_EXPLORE → org is building in the same space as the seller and could be an
                  integration or ecosystem partner rather than a direct customer
                  OR org is a platform player whose end-users are the real prospects

SKIP            → confidence < 35 AND no plausible seller fit
                  OR org is a direct competitor with no integration angle
                  OR insufficient public data to make any judgement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT QUALITY STANDARDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evidence items MUST follow this format exactly:
- PR: "<exact PR title>" in repo '<repo_name>'
- Issue Label: "<label>" found in '<repo_name>'
- Repo Topic: '<repo_name>' tagged with '<topic>'
- Repo: '<repo_name>' — <brief observation>
- Commit: "<exact commit message>" in '<repo_name>'

DO NOT write:
- "They are improving their infrastructure." (vague)
- "Active development detected." (meaningless)
- "They may need your product." (not evidence)

TONE: You are a premium sales intelligence platform. Be specific, direct, analytical. No fluff.
"""

    prompt = f"""
ANALYZE THIS ORGANIZATION FOR SALES INTELLIGENCE:

━━ ORGANIZATION ━━
Name: {payload['org']}
Industry: {payload['industry']}
Business Context: {payload['business_context']}
Seller's Product: {payload.get('seller_product', 'Not specified')}

━━ REPOSITORY OVERVIEW ({len(payload['repos'])} repos analyzed) ━━
{json.dumps(payload['repos'], indent=2)}

━━ ENGINEERING SIGNALS ━━
Repository Topics: {payload['topics']}
Primary Languages: {payload['languages']}
Active Issue Labels: {payload['issue_labels']}
CI/CD Tools Detected: {payload.get('workflow_tools', [])}

━━ PULL REQUEST SIGNALS (highest priority) ━━
{json.dumps(payload['pr_titles'], indent=2)}

━━ OPEN ISSUE SIGNALS ━━
{json.dumps(payload['issue_titles'], indent=2)}

━━ KEYWORD SIGNAL MAP ━━
User-Defined Keyword Hits: {payload['user_custom_hits']}
Intent Category Hits: {json.dumps(
    {k: v for k, v in payload['category_hits'].items() if v and k != 'user_custom'},
    indent=2
)}

━━ ACTIVITY METRICS ━━
Active Contributors: {payload['contributors_active']}
Total Commits Analyzed: {payload['total_commits']}
Activity Level: {payload['activity_level']}
Algorithmic Signal Score: {payload['algorithmic_score']}/100

━━ RECENT COMMITS (supporting evidence only) ━━
{json.dumps(payload['recent_commits'][:10], indent=2)}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.05,  # Very low — factual output, not creative
                response_mime_type="application/json",
                response_schema=AgentInsight,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Vertex AI error: {e}")
        return _fallback_insight()


def _fallback_insight() -> dict:
    return {
        "signal": "Analysis Unavailable",
        "summary": "AI analysis could not be completed. Raw metrics are available below.",
        "evidence": ["Fallback mode active — check Vertex AI configuration."],
        "business_insight": "Unable to generate insight — raw GitHub metrics are available in raw_metrics.",
        "opportunity": "Review configuration and retry.",
        "urgency": "Low",
        "confidence": 0,
        "activity_level": "Unknown",
        "tech_stack_signals": [],
        "intent_category": "unknown",
        "is_product_builder": False,
        "recommended_action": "SKIP",
    }


# ==========================================
# 8. MAIN ENDPOINT
# ==========================================

@app.post("/api/analyze")
async def analyze_github_intent(request: AnalyzeRequest):
    start_time = time.time()
    org = extract_org_name(request.github_org)
    logger.info(f"[{org}] Analysis started")

    try:
        async with httpx.AsyncClient() as client:

            # ── Step 1: Fetch org repos ────────────────────────────────────────
            # Fetch top 20 by last-updated, then keep the 15 most relevant non-forks.
            # GITHUB_SEMAPHORE (12 slots) handles concurrency safely — 75 requests
            # queue and process 12 at a time. No need to reduce repo count; more
            # repos = better AI grounding, which is the higher priority.
            repos_url = (
                f"https://api.github.com/orgs/{org}"
                f"/repos?sort=updated&per_page=20&type=public"
            )
            repos_raw = await fetch_github_data_async(repos_url, client)

            if not repos_raw or not isinstance(repos_raw, list):
                logger.warning(f"[{org}] Org not found or no public repos")
                return {
                    "status": "error",
                    "message": f"GitHub org '{org}' not found or has no public repositories.",
                }

            # Exclude forks — they represent maintenance activity, not org intent.
            # 15 repos × 5 endpoints = 75 requests, processed safely 12 at a time
            # by GITHUB_SEMAPHORE. The semaphore alone prevents connection storms —
            # reducing repo count further would only sacrifice AI grounding quality.
            owned_repos = [r for r in repos_raw if not r.get("fork", False)][:20]
            logger.info(
                f"[{org}] {len(repos_raw)} repos fetched, "
                f"{len(owned_repos)} non-forks selected for analysis"
            )

            enriched_repos, all_topics, language_counts = extract_repo_metadata(owned_repos)

            # ── Step 2: Fetch rich signals concurrently per repo ───────────────
            tasks = []
            task_map = []

            for repo in owned_repos:
                rname = repo.get("name")
                if not rname:
                    continue

                base = f"https://api.github.com/repos/{org}/{rname}"

                tasks.append(fetch_github_data_async(f"{base}/commits?per_page=15", client))
                task_map.append(("commits", rname))

                tasks.append(fetch_github_data_async(f"{base}/contributors?per_page=100", client))
                task_map.append(("contributors", rname))

                # Pull Requests — richest signal source
                tasks.append(fetch_github_data_async(
                    f"{base}/pulls?state=all&sort=updated&per_page=15", client
                ))
                task_map.append(("prs", rname))

                # Issues with labels — engineering priorities
                tasks.append(fetch_github_data_async(
                    f"{base}/issues?state=open&per_page=15", client
                ))
                task_map.append(("issues", rname))

                # Releases — product cadence signal
                tasks.append(fetch_github_data_async(f"{base}/releases?per_page=5", client))
                task_map.append(("releases", rname))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_commits: list = []
            all_prs: list = []
            all_issues: list = []
            all_releases: list = []
            unique_contributors: set = set()

            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"[{org}] Async gather error at index {i}: {res}")
                    continue
                if not isinstance(res, list):
                    continue

                task_type, rname = task_map[i]

                if task_type == "commits":
                    for c in res:
                        c["_repo"] = rname
                    all_commits.extend(res)

                elif task_type == "contributors":
                    for u in res:
                        if isinstance(u, dict) and u.get("login"):
                            unique_contributors.add(u["login"])

                elif task_type == "prs":
                    for pr in res:
                        pr["_repo"] = rname
                    all_prs.extend(res)

                elif task_type == "issues":
                    # GitHub's /issues endpoint also returns PRs — filter them out
                    for issue in res:
                        if "pull_request" not in issue:
                            issue["_repo"] = rname
                            all_issues.append(issue)

                elif task_type == "releases":
                    all_releases.extend(res)

        # ── Step 3: Extract and clean signals ─────────────────────────────────
        pr_titles = extract_pr_signals(all_prs)
        issue_titles_raw, issue_labels = extract_issue_signals(all_issues)

        # Clean commits — remove noise, deduplicate, tag with repo name
        clean_commits = []
        seen_messages: set = set()
        for c in all_commits:
            msg = c.get("commit", {}).get("message", "").split("\n")[0].strip().lower()
            msg_clean = re.sub(r"\(#\d+\)", "", msg).strip()
            if (
                msg_clean in seen_messages
                or any(n in msg_clean for n in NOISE_COMMITS)
                or len(msg_clean) < 15
            ):
                continue
            seen_messages.add(msg_clean)
            clean_commits.append({
                "repo": c.get("_repo", ""),
                "message": c.get("commit", {}).get("message", "").split("\n")[0][:100],
            })

        # ── Step 4: Score signals ──────────────────────────────────────────────
        category_hits, algo_score = score_signals(
            user_kws=request.custom_keywords,
            enriched_repos=enriched_repos,
            commits=all_commits,
            pr_titles=pr_titles,
            issue_titles=issue_titles_raw,
            issue_labels=issue_labels,
            topics=all_topics,
        )

        # ── Step 5: Activity level ─────────────────────────────────────────────
        activity_score = (
            len(unique_contributors) * 1.0
            + len(owned_repos) * 2
            + len(all_commits) * 0.2
            + len(all_prs) * 0.5
        )
        activity_level = (
            "High" if activity_score > 25
            else "Medium" if activity_score > 12
            else "Low"
        )

        # ── Step 6: Build AI payload ───────────────────────────────────────────
        ai_payload = {
            "org": org,
            "industry": request.industry,
            "business_context": request.business_context,
            "seller_product": request.seller_product,
            "repos": enriched_repos,
            "topics": all_topics[:30],
            "languages": dict(sorted(language_counts.items(), key=lambda x: -x[1])[:8]),
            "issue_labels": issue_labels,
            "pr_titles": pr_titles,
            "issue_titles": issue_titles_raw,
            "workflow_tools": [],
            "user_custom_hits": category_hits.get("user_custom", []),
            "category_hits": category_hits,
            "contributors_active": len(unique_contributors),
            "total_commits": len(all_commits),
            "recent_commits": clean_commits[:15],
            "activity_level": activity_level,
            "algorithmic_score": algo_score,
        }

        # Run synchronous AI call in a thread — keeps the event loop unblocked
        agent_insight = await asyncio.to_thread(call_intent_agent, ai_payload)

        duration = time.time() - start_time
        logger.info(
            f"[{org}] Completed in {duration:.2f}s | "
            f"Score: {algo_score} | "
            f"Confidence: {agent_insight.get('confidence', '?')} | "
            f"Action: {agent_insight.get('recommended_action', '?')}"
        )

        return {
            "status": "success",
            "org_analyzed": org,
            "analysis_duration_s": round(duration, 2),
            "raw_metrics": {
                "repos_analyzed": len(owned_repos),
                "prs_analyzed": len(all_prs),
                "issues_analyzed": len(all_issues),
                "contributors_active": len(unique_contributors),
                "topics_found": all_topics[:20],
                "languages": dict(list(language_counts.items())[:6]),
                "issue_labels_found": issue_labels,
                "user_keywords_matched": category_hits.get("user_custom", []),
                "intent_categories_matched": {
                    k: v for k, v in category_hits.items()
                    if v and k != "user_custom"
                },
                "pre_ai_algorithmic_score": algo_score,
                "activity_level": activity_level,
            },
            "agent_insight": agent_insight,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"[{org}] Fatal error after {duration:.2f}s: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "org_analyzed": org,
            "analysis_duration_s": round(duration, 2),
            "message": str(e),
            "agent_insight": _fallback_insight(),
        }


# ==========================================
# 9. HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.0",
        "vertex_ai": VERTEX_AVAILABLE,
        "cache_entries": len(CACHE),
        "semaphore_slots_free": GITHUB_SEMAPHORE._value,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)