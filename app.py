import os
import re
import json
import logging
import time
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple

from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & LOGGING CONFIGURATION
# ==========================================

# Issue 10: Richer Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GitLeads-Advanced")

app = FastAPI(title="GitLeads Intent Intelligence API", version="2.0")

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-phishguard-project-id")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

try:
    ai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    VERTEX_AVAILABLE = True
    logger.info("Vertex AI Client Initialized Successfully.")
except Exception as e:
    logger.error(f"Vertex AI Init Error: {e}")
    VERTEX_AVAILABLE = False

DEFAULT_KEYWORDS = ["kubernetes", "docker", "auth", "infra", "ci", "security", "pipeline", "migration", "aws",
                    "terraform"]

FALLBACK_DATA = {
    "status": "success (fallback)",
    "agent_insight": {
        "signal": "Active Engineering Developments Detected",
        "evidence": ["Fallback mode activated.", "Raw GitHub metrics indicate activity."],
        "business_insight": "The organization is actively committing code, but detailed AI analysis was safely bypassed.",
        "opportunity": "Verify system configuration.",
        "urgency": "Low",
        "confidence": 50,
        "activity_level": "Medium"
    }
}

# Issue 5: Simple In-Memory Cache
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes TTL


# ==========================================
# 2. DATA MODELS
# ==========================================

class AnalyzeRequest(BaseModel):
    github_org: str
    custom_keywords: List[str] = Field(default_factory=list)
    business_context: str = "General B2B SaaS"
    industry: str = "General"


class AgentInsight(BaseModel):
    signal: str = Field(description="Short, clear name of the detected signal")
    evidence: List[str] = Field(description="Formatted list of key observations.")
    business_insight: str = Field(description="Business meaning and intent.")
    opportunity: str = Field(description="Opportunity for the user based on context.")
    urgency: str = Field(description="High, Medium, or Low")
    confidence: int = Field(description="0 to 100 percentage score.")
    activity_level: str = Field(description="High, Medium, or Low.")


# ==========================================
# 3. GITHUB FETCHING (ASYNC & CACHED)
# ==========================================

# Issue 2: Regex extraction for stability
def extract_org_name(github_input: str) -> str:
    match = re.search(r"github\.com/([^/?]+)", github_input)
    if match:
        return match.group(1).strip()
    return github_input.strip().strip("/")


# Issue 3, 4, 5: Async fetching with caching and robust rate-limit handling
async def fetch_github_data_async(url: str, client: httpx.AsyncClient) -> Any:
    # Check Cache
    if url in CACHE and time.time() - CACHE[url]['timestamp'] < CACHE_TTL:
        return CACHE[url]['data']

    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    for attempt in range(3):
        try:
            response = await client.get(url, headers=headers, timeout=10.0)

            # Rate limit handling for both 403 and 429
            if response.status_code in (403, 429):
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_time = max(reset_time - time.time(), 1)
                logger.warning(f"Rate limited (Status {response.status_code}). Sleeping for {sleep_time} seconds.")
                await asyncio.sleep(min(sleep_time, 5))
                continue

            if response.status_code == 200:
                data = response.json()
                CACHE[url] = {'data': data, 'timestamp': time.time()}
                return data
            return []  # Return empty on 404 or other errors to avoid crashing orchestration

        except httpx.RequestError as e:
            logger.error(f"HTTP Request error for {url}: {e}")
            await asyncio.sleep(2)
            continue

    return []


# ==========================================
# 4. SIGNAL FILTERING & PREPROCESSING
# ==========================================

def filter_and_score_signals(commits_data: list, repos: list, user_keywords: list) -> Tuple[
    List[str], List[str], List[str], List[str], int]:
    user_matched_commits, default_matched_commits = [], []
    user_hits, default_hits = set(), set()
    seen_commits = set()

    ignore_list = {"message", "update", "fix", "test", "add", "remove", "bump", "version"}
    cleaned_user_kws = [k.lower().strip() for k in user_keywords if
                        k.strip() and len(k.strip()) >= 2 and k.lower().strip() not in ignore_list]
    noise_words = ["merge pull request", "update readme", "typo", "chore", "version packages", "bump", "lint",
                   "cleanup", "release", ".gitignore"]

    # Issue 7: Keyword scoring may miss intent in repo names
    for repo in repos:
        repo_name = repo.get("name", "").lower()
        for kw in cleaned_user_kws:
            if re.search(rf"\b{re.escape(kw)}\b", repo_name):
                user_hits.add(kw)
        for kw in DEFAULT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", repo_name):
                default_hits.add(kw)

    for commit in commits_data:
        msg = commit.get("commit", {}).get("message", "").lower()
        if any(noise in msg for noise in noise_words):
            continue

        clean_msg = msg.split('\n')[0].strip()
        clean_msg = re.sub(r'\(#\d+\)', '', clean_msg).strip()

        if clean_msg in seen_commits:
            continue
        seen_commits.add(clean_msg)

        matched_user, matched_default = False, False

        for kw in cleaned_user_kws:
            if re.search(rf"\b{re.escape(kw)}\b", msg):
                user_hits.add(kw)
                matched_user = True

        for kw in DEFAULT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", msg):
                default_hits.add(kw)
                matched_default = True

        if matched_user and len(user_matched_commits) < 10:
            user_matched_commits.append(clean_msg)
        elif matched_default and len(default_matched_commits) < 10:
            default_matched_commits.append(clean_msg)

    base_score = min(100, (len(user_hits) * 15) + (len(default_hits) * 5) + (len(user_matched_commits) * 10))

    return user_matched_commits, default_matched_commits, list(user_hits), list(default_hits), int(base_score)


# ==========================================
# 5. VERTEX AI AGENT
# ==========================================

def call_intent_agent(payload: dict) -> dict:
    if not VERTEX_AVAILABLE:
        return FALLBACK_DATA["agent_insight"]

    system_instruction = """
    You are an expert B2B Sales Intelligence Analyst specializing in interpreting engineering activity as business buying signals.

    Your job is to analyze GitHub repository activity and translate it into clear, concise, and actionable business intelligence for sales, product, and growth teams.

    You do NOT describe code. You do NOT summarize commits.

    Instead, you infer:
    - What the company is currently doing
    - Why they are doing it
    - What tools, services, or solutions they are likely to need next

    You think in terms of:
    - Infrastructure changes
    - Technology adoption
    - Team growth
    - Product evolution
    - Security posture
    - Scaling signals

    You must ALWAYS produce structured, high-quality output that is:
    - Specific (not generic)
    - Business-relevant (not technical fluff)
    - Concise (no long paragraphs)
    - Insightful (clear reasoning)

    Avoid vague statements like:
    “They are improving their system” or “They may need tools”

    Be confident and analytical.

    If signals are weak or unclear, say so and lower confidence instead of guessing.

    CRITICAL GROUNDING RULES:
    1. You MUST USE ONLY the commits, repos, keywords, and metrics provided in the GITHUB SIGNAL DATA payload.
    2. DO NOT invent, assume, or hallucinate repositories, technologies, contributors, or timelines.
    3. Do NOT predict future product launches unless explicitly supported by evidence. Prefer terms like "suggests" over "likely will".
    4. Always enclose repository names in single quotes (e.g., 'next.js') for consistency.

    CONFIDENCE SCORING RULES:
    - 80-100: Strong multi-signal evidence (repo activity + multiple commits + high keyword density).
    - 50-79: Moderate evidence (few commits, partial keyword match).
    - <50: Weak or sparse signals. Lower confidence strictly if data is lacking.

    EVIDENCE FORMATTING RULES:
    Strictly format your 'evidence' array strings like this:
    - Commit: "<exact commit message>"
    - Repo Activity: "'<repo_name>' actively updated"

    🧠 Intelligence Rules (VERY IMPORTANT)
    Follow these reasoning principles:

    1. If infrastructure-related keywords appear (e.g., kubernetes, docker, infra):
    → Interpret as scaling, modernization, or architecture change.

    2. If contributor count increases significantly:
    → Interpret as team growth or active development phase.

    3. If new repositories are created with names like "platform", "core", "infra":
    → Interpret as new initiative or investment area.

    4. If security-related changes appear:
    → Interpret as strengthening security posture or compliance effort.

    5. If user-defined keywords match strongly:
    → PRIORITIZE those signals in your analysis.

    6. Always connect technical activity → business intent.

    7. If multiple weak signals exist:
    → Combine them into one reasonable hypothesis.

    8. If signals are insufficient:
    → Output a LOW confidence insight instead of guessing.

    ⚡ Optional Enhancement (Advanced Layer)
    Also ensure the tone sounds like a premium SaaS intelligence platform, not an AI assistant.
    """

    prompt_content = f"""
    TARGET COMPANY: {payload['org']}
    INDUSTRY: {payload['industry']}
    BUSINESS CONTEXT: {payload['business_context']}

    USER KEYWORDS FOUND: {payload['user_keyword_hits']}
    DEFAULT KEYWORDS FOUND: {payload['default_keyword_hits']}

    EVIDENCE BLOCKS:
    User Matches: {json.dumps(payload['user_matched_commits'])}
    Default Matches: {json.dumps(payload['default_matched_commits'])}
    Active Repos: {json.dumps(payload['repos'])}

    METRICS:
    Contributor Trend: {payload['activity_level']}
    Algorithmic Signal Strength: {payload['algorithmic_score']}/100
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=AgentInsight
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Vertex AI Error: {e}")
        return FALLBACK_DATA["agent_insight"]


# ==========================================
# 6. MAIN ENDPOINT
# ==========================================

@app.post("/api/analyze")
async def analyze_github_intent(request: AnalyzeRequest):
    start_time = time.time()
    org = extract_org_name(request.github_org)

    logger.info(f"Analysis started: {org}")

    try:
        async with httpx.AsyncClient() as client:
            repos_url = f"https://api.github.com/orgs/{org}/repos?sort=updated&per_page=5"
            repos = await fetch_github_data_async(repos_url, client)

            # Issue 8: Validation of org existence
            if not repos:
                logger.warning(f"GitHub organization not found or empty: {org}")
                return {"status": "error", "message": "GitHub organization not found or has no public repositories"}

            all_commits = []
            unique_contributors = set()
            repo_names = []

            tasks = []
            task_mapping = []

            for repo in repos:
                repo_name = repo.get("name")
                if not repo_name: continue
                repo_names.append(repo_name)

                commits_url = f"https://api.github.com/repos/{org}/{repo_name}/commits?per_page=10"
                contributors_url = f"https://api.github.com/repos/{org}/{repo_name}/contributors?per_page=100"

                # Queue up concurrent tasks (Issue 3)
                tasks.append(fetch_github_data_async(commits_url, client))
                task_mapping.append(('commits', repo_name))

                tasks.append(fetch_github_data_async(contributors_url, client))
                task_mapping.append(('contributors', repo_name))

            # Execute all network calls concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"Error fetching async data: {res}")
                    continue

                task_type, _ = task_mapping[i]
                if task_type == 'commits' and isinstance(res, list):
                    all_commits.extend(res)
                elif task_type == 'contributors' and isinstance(res, list):
                    for user in res:
                        if isinstance(user, dict) and user.get("login"):
                            unique_contributors.add(user.get("login"))

        u_commits, d_commits, u_hits, d_hits, algo_score = filter_and_score_signals(all_commits, repos,
                                                                                    request.custom_keywords)

        # Issue 6: More realistic contributor scoring
        activity_score = len(unique_contributors) + (len(repo_names) * 2) + (len(all_commits) / 5)
        if activity_score > 20:
            activity_level = "High"
        elif activity_score > 10:
            activity_level = "Medium"
        else:
            activity_level = "Low"

        ai_payload = {
            "org": org,
            "industry": request.industry,
            "business_context": request.business_context,
            "user_keyword_hits": u_hits,
            "default_keyword_hits": d_hits,
            "user_matched_commits": u_commits,
            "default_matched_commits": d_commits,
            "repos": repo_names,
            "activity_level": activity_level,
            "algorithmic_score": algo_score
        }

        # Prevent blocking the event loop on the synchronous AI call
        agent_insight = await asyncio.to_thread(call_intent_agent, ai_payload)

        # Issue 10: Duration logging
        duration = time.time() - start_time
        logger.info(f"Analysis completed for {org} in {duration:.2f}s")

        return {
            "status": "success",
            "org_analyzed": org,
            "raw_metrics": {
                "user_keywords_found": u_hits,
                "default_keywords_found": d_hits,
                "contributors_active": len(unique_contributors),
                "pre_ai_algorithmic_score": algo_score
            },
            "agent_insight": agent_insight
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Extraction Error for {org} after {duration:.2f}s: {e}")
        return FALLBACK_DATA


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080)