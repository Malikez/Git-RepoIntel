import os
import requests
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# ---> NEW UNIFIED GOOGLE GEN-AI SDK (Configured for Vertex)
from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

app = FastAPI(title="GitLeads Intent Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ---> VERTEX AI INITIALIZATION
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-phishguard-project-id")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

try:
    # This uses the modern unified SDK but routes securely through your GCP Vertex infrastructure
    ai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    VERTEX_AVAILABLE = True
except Exception as e:
    print(f"Vertex AI Init Error (Check Credentials): {e}")
    VERTEX_AVAILABLE = False

DEFAULT_KEYWORDS = ["kubernetes", "docker", "auth", "infra", "ci", "security", "pipeline", "migration", "aws",
                    "terraform"]

FALLBACK_DATA = {
    "status": "success (fallback)",
    "score": 10,
    "agent_insight": {
        "signal": "Active Engineering Developments Detected",
        "evidence": [
            "Fallback mode activated due to API limits or missing credentials.",
            "Raw GitHub metrics indicate recent repository activity."
        ],
        "business_insight": "The organization is actively committing code, but detailed AI analysis was safely bypassed. Please check API credentials.",
        "opportunity": "Verify system configuration.",
        "urgency": "Low",
        "confidence": 50,
        "activity_level": "Medium"
    }
}


# ==========================================
# 2. DATA MODELS & HELPERS
# ==========================================

class AnalyzeRequest(BaseModel):
    github_org: str
    custom_keywords: List[str] = Field(default_factory=list)
    business_context: str = "General B2B SaaS"
    industry: str = "General"


class AgentInsight(BaseModel):
    signal: str = Field(description="Short, clear name of the detected signal")
    evidence: List[str] = Field(
        description="List of 2-3 key observations extracted STRICTLY from the provided GitHub payload data.")
    business_insight: str = Field(description="2-3 sentences explaining business meaning and intent")
    opportunity: str = Field(
        description="What type of product/service/vendor could benefit based on the user's business context")
    urgency: str = Field(description="High, Medium, or Low")
    confidence: int = Field(description="0 to 100 percentage. Lower this if data is sparse.")
    activity_level: str = Field(description="High, Medium, or Low based on the active contributor count")


def extract_org_name(github_input: str) -> str:
    github_input = github_input.strip()
    if "github.com/" in github_input:
        parts = github_input.split("github.com/")
        if len(parts) > 1:
            return parts[1].strip("/").split("/")[0]
    return github_input.strip("/")


# ==========================================
# 3. GITHUB FETCHING & EXTRACTION LOGIC
# ==========================================

def get_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def fetch_top_repos(org: str, limit: int = 5):
    url = f"https://api.github.com/orgs/{org}/repos?sort=updated&per_page={limit}"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()


def fetch_commits(org: str, repo: str, limit: int = 10):
    url = f"https://api.github.com/repos/{org}/{repo}/commits?per_page={limit}"
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        return []
    return response.json()


def fetch_contributors(org: str, repo: str) -> set:
    url = f"https://api.github.com/repos/{org}/{repo}/contributors?per_page=100"
    response = requests.get(url, headers=get_headers())
    if response.status_code != 200:
        return set()
    return {user.get("login") for user in response.json() if user.get("login")}


def detect_activity_level(count: int):
    if count > 15:
        return "High"
    elif count > 5:
        return "Medium"
    return "Low"


def filter_commits_and_separate_keywords(commits_data: list, user_keywords: list):
    user_matched_commits = []
    default_matched_commits = []
    user_hits = set()
    default_hits = set()

    ignore_list = {"message", "update", "fix", "test", "add", "remove", "bump", "version"}
    cleaned_user_kws = [k.lower().strip() for k in user_keywords if
                        k.strip() and len(k.strip()) >= 2 and k.lower().strip() not in ignore_list]
    noise_words = ["merge pull request", "update readme", "typo", "chore", "version packages", "bump", "lint",
                   "cleanup", "release", ".gitignore"]

    for commit in commits_data:
        msg = commit.get("commit", {}).get("message", "").lower()
        if any(noise in msg for noise in noise_words):
            continue

        clean_msg = msg.split('\n')[0]
        clean_msg = re.sub(r'\(#\d+\)', '', clean_msg).strip()

        matched_user = False
        matched_default = False

        # Check User Keywords First
        for kw in cleaned_user_kws:
            if re.search(rf"\b{re.escape(kw)}\b", msg):
                user_hits.add(kw)
                matched_user = True

        # Check Default Keywords
        for kw in DEFAULT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", msg):
                default_hits.add(kw)
                matched_default = True

        # Append to respective lists (prioritizing user matches)
        if matched_user and clean_msg not in user_matched_commits:
            user_matched_commits.append(clean_msg)
        elif matched_default and clean_msg not in default_matched_commits:
            default_matched_commits.append(clean_msg)

    # Return top 5 of each to avoid token bloat
    return user_matched_commits[:5], default_matched_commits[:5], list(user_hits), list(default_hits)


# ==========================================
# 4. VERTEX AI AGENT (THE GROUNDED BRAIN)
# ==========================================

def call_intent_agent(payload: dict) -> dict:
    if not VERTEX_AVAILABLE:
        print("WARNING: Vertex AI not initialized. Using fallback data.")
        return FALLBACK_DATA["agent_insight"]

    # 1. FULL MASTER AI SYSTEM PROMPT (Restored verbatim + New Guardrails)
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

    # 2. FULL DYNAMIC INPUT TEMPLATE
    prompt_content = f"""
    TARGET COMPANY:
    {payload['org']}

    INDUSTRY CONTEXT:
    {payload['industry']}

    USER-DEFINED KEYWORDS:
    {', '.join(payload['custom_keywords']) if payload['custom_keywords'] else 'None provided'}

    DEFAULT SIGNAL KEYWORDS:
    {', '.join(DEFAULT_KEYWORDS)}

    BUSINESS CONTEXT (IF PROVIDED):
    {payload['business_context']}

    GITHUB SIGNAL DATA (FILTERED):
    User Keyword Matched Commits: {json.dumps(payload['user_matched_commits'], indent=2)}
    Default Keyword Matched Commits: {json.dumps(payload['default_matched_commits'], indent=2)}
    Active Repositories Scanned: {json.dumps(payload['repos'], indent=2)}

    ADDITIONAL SIGNAL METRICS:
    - Number of recent user-matched commits: {len(payload['user_matched_commits'])}
    - Number of recent default-matched commits: {len(payload['default_matched_commits'])}
    - Total active repos scanned: {payload['repo_count']}
    - Contributor trend (Activity Level): {payload['activity_level']}
    """

    try:
        # 3. Execution using Pydantic Strict Schema & Unified SDK
        response = ai_client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,  # Extremely low temp forces adherence to provided data
                response_mime_type="application/json",
                response_schema=AgentInsight  # Uses the Pydantic class to guarantee shape
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Vertex AI Execution Error: {e}")
        return FALLBACK_DATA["agent_insight"]


# ==========================================
# 5. MAIN ENDPOINT
# ==========================================

@app.get("/")
def health_check():
    return {"status": "online", "message": "GitLeads AI Engine is active on Vertex AI (Unified SDK)."}


@app.post("/api/analyze")
def analyze_github_intent(request: AnalyzeRequest):
    org = extract_org_name(request.github_org)

    try:
        repos = fetch_top_repos(org, limit=5)
        all_commits = []
        unique_contributors = set()
        repo_names = []

        for repo in repos:
            repo_names.append(repo["name"])
            commits = fetch_commits(org, repo["name"], limit=10)
            all_commits.extend(commits)
            unique_contributors.update(fetch_contributors(org, repo["name"]))

        # The new separated keyword extraction
        user_matched_commits, default_matched_commits, user_hits, default_hits = filter_commits_and_separate_keywords(
            all_commits, request.custom_keywords)
        activity_level = detect_activity_level(len(unique_contributors))

        # Build the dynamic, deeply grounded payload
        ai_payload = {
            "org": org,
            "industry": request.industry,
            "business_context": request.business_context,
            "custom_keywords": request.custom_keywords,
            "user_keyword_hits": user_hits,
            "default_keyword_hits": default_hits,
            "user_matched_commits": user_matched_commits,
            "default_matched_commits": default_matched_commits,
            "repos": repo_names,
            "repo_count": len(repos),
            "activity_level": activity_level
        }

        # Call Vertex AI
        agent_insight = call_intent_agent(ai_payload)

        return {
            "status": "success",
            "org_analyzed": org,
            "raw_metrics": {
                "user_keywords_found": user_hits,
                "default_keywords_found": default_hits,
                "contributors_active": len(unique_contributors)
            },
            "agent_insight": agent_insight
        }

    except Exception as e:
        print(f"Extraction Error: {e}")
        return FALLBACK_DATA


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)