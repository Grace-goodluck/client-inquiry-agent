# Client Inquiry Agent by TechByGrace

Paste one client message and get back:

1. **Category**: `new_project`, `pricing`, `timeline`, `policy`, `support` or `other`.
2. **Draft reply** in Grace's voice: warm, professional, signed "Grace, TechByGrace".
3. **Sources**: which brand file (and section) each fact came from.
4. **Needs Grace**: questions the brand files don't answer, for Grace to handle herself.

It **drafts only and never sends anything**. Grace reviews every reply.

## How it works

- **Knowledge:** `public_brand/` holds copies of Grace's bio, services, how-I-work, FAQ and portfolio files, with every price figure removed. Grace reviews this folder before it is committed. Locally, `--brain` uses the Second Brain API at `http://127.0.0.1:8000` instead. The web demo always uses `public_brand/`.
- **Retrieval:** BM25 keyword search in plain Python (no embeddings). It retrieves the top 5 chunks, with plural normalisation and a few query-only synonyms (Saturday/Sunday → weekend, cost/price → pricing, and so on). For vague messages, the how-I-work steps and the services overview are added.
- **One Gemini call per message** (default `gemini-3.6-flash`). Gemini returns JSON: category, reply, the ids of the sources it used, and needs_grace.

## Safety rules (enforced in code)

- **Price guard:** any money amount in a draft (₦, $, NGN, USD, "50k", "1.5m", ranges, numbers like 150,000) is replaced with "Pricing depends on scope. I'll send you a quote after a short call." Percentages such as "50% upfront" are allowed, because they are payment terms, not prices.
- **Number check:** every number in the draft must appear in the retrieved brand text. Otherwise the draft is replaced with a safe fallback ("…I'll reply to this personally soon. — Grace, TechByGrace") and the question goes to Needs Grace.
- **Facts only from the files:** Gemini is told to use only the retrieved facts and to put anything else in Needs Grace. Source ids it returns that weren't given are dropped and reported.
- **Strict output:** broken JSON or an unknown category is an error, never a guess.
- **Web demo:** "Use made-up messages only. Don't enter personal details." Messages are never stored or logged; logs record only the category and timing. Limits: 5 drafts per hour per visitor, 20 per day in total, 2,000 characters per message.
- **Gemini:** 60 s timeout; 503, 504 and dropped connections are retried (command line: 30 s, then 60 s; web: once after 10 s); 429 is never retried. The API key comes from the `GEMINI_API_KEY` environment variable only; no `.env` file is ever loaded.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and a Gemini API key.

```powershell
uv sync --extra web
$env:GEMINI_API_KEY = "your-key"            # optional: $env:CLIENT_AGENT_MODEL = "gemini-3.6-flash"
```

Command line:

```powershell
uv run client-inquiry "Hi Grace, do you work on weekends?"
uv run client-inquiry --file message.txt --json
uv run client-inquiry "Do you offer refunds?" --brain    # uses the local Second Brain API
```

Web page (then open http://127.0.0.1:8000):

```powershell
uv run client-inquiry-web
```

Tests (no real Gemini calls):

```powershell
uv run pytest
```

Docker:

```powershell
docker build -t client-inquiry-agent .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=your-key client-inquiry-agent
```

## Deployment

Docker image on the Render free plan. Render sets `PORT`; `GEMINI_API_KEY` (from the project's own Google project, `techbygrace-inquiry-demo`) is set as a secret environment variable in the Render dashboard. `/health` answers GET and HEAD for Render's health check and the UptimeRobot monitor.

## Evaluation

`evaluation/inquiries.md` holds 10 made-up inquiries, each with an expected category, facts that must appear and phrases that must not appear. They include weekends, refunds, code ownership, a price question, a new project, a timeline question, support after delivery, a question the files can't answer, a mixed message and a vague one.

```powershell
uv run python scripts/run_evaluation.py              # all 10 (real Gemini calls, about 13 s apart)
uv run python scripts/run_evaluation.py --only E01,E05
```

Results are written to `evaluation/results-<date>.md`, with a verdict line under each draft for Grace.

### Results

| Run | Passed every automatic check |
|---|---|
| First run (2026-11-09) | 6/10 |
| Final run (2026-11-11) | **10/10** ([evaluation/results-2026-11-11.md](evaluation/results-2026-11-11.md)) |

**First run:** the failures were search gaps (for example, "Saturday" didn't match "weekends") plus one Gemini timeout. There were no price, invented-number or policy failures.

**Fixes:** a query synonym list, fallback facts for vague messages, top_k raised from 4 to 5, and clearer source-id matching.

**Final run:** all 10 inquiries had the expected category, no money amounts, and every required fact present.

**Honest note:** the fixes were guided by these same 10 inquiries, so the final run shows those gaps were closed, not general accuracy. A larger, unseen test set is future work.

## Out of scope for v1

Sending email, conversation memory, booking calls.
