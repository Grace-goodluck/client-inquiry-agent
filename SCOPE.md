# Client Inquiry Agent — Scope

Decisions recorded: 2026-11-09

## Purpose

The user pastes one client message. The agent returns:

1. **category**: one of `new_project`, `pricing`, `timeline`, `policy`, `support`, `other`.
2. **draft reply** in Grace's voice: warm and professional, signed "Grace, TechByGrace".
3. **sources**: which brand file each fact came from.
4. **needs_grace**: questions it could not answer from the files.

The agent only writes drafts. It never sends anything.

## Knowledge

- **Live demo:** a `public_brand/` folder in the repo. It holds copies of Grace's bio, services, how-I-work, FAQ and portfolio files, with all price figures removed. Grace reviews this folder before it is committed. The lessons note is not included.
- **Local:** an optional `--brain` flag uses the Second Brain API at `http://127.0.0.1:8000` instead.
- **Retrieval:** BM25 keyword search over `public_brand/` chunks. No embedding calls. A trailing plural "s"/"es" is dropped from words in both queries and chunks, so "refund" matches "refunds".

### Retrieval and reply decisions (2026-11-09, after the first evaluation run)

The first run failed E01, E05 and E10 because the client's words didn't match Grace's ("Saturday/Sunday" vs "weekends", "get started" vs "start", and "hi, interested" matched nothing). E03 also came back with no sources.

- **(A) Query synonyms, applied to the query only, never to the chunks:** Saturday/Sunday → weekend; started/starting → start; cost/price/charge/quote → pricing; delivery/deliver/delivered/launch/launched → support. The original word is kept and the synonym is searched too.
- **(B) Fallback chunks:** when the best BM25 score is below 2.0, or nothing matches, the how-I-work process steps (1–7) and the services overview (bio "What I do") are added after whatever BM25 found. Real evaluation questions score 3.3 or more at the top; greetings score 0. A test keeps every evaluation question except the vague one above the threshold. Applies to `public_brand/` only, not to `--brain`.
- **(E) top_k raised from 4 to 5.** E05's proposal chunk ("2. Written proposal and quote") ranked 5th after the synonyms, so with 4 chunks Gemini never saw it. Every message now retrieves the top 5 BM25 chunks (`TOP_K` in config.py).
- **(C) Vague messages:** the Gemini prompt tells it that if a message is vague and doesn't say what the client needs, the reply should warmly ask what they need.
- **(D) Source ids:** the ids Gemini returns are matched to the given ids leniently. Case, extra spaces, surrounding quotes or brackets, an "id:" prefix and trailing punctuation are ignored. Ids that still match nothing are kept in `dropped_sources` and shown in the CLI notes, the `--json` output and the evaluation report, instead of disappearing. The prompt also says to copy ids without brackets and never to leave sources empty when the reply states a fact. The exact cause of the empty E03 sources is unknown, because the first run didn't record what Gemini returned; the next run will show it.

## Safety (enforced in code)

- **Price guard:** any money amount in a draft (₦, $, NGN, "50k"-style amounts) is replaced with:
  > Pricing depends on scope. I'll send you a quote after a short call.

  Percentages such as "50% upfront" are allowed, because they are payment terms, not prices.
  - "N250,000" (a capital N directly followed by digits) counts as a naira amount and is replaced.
  - Bare shorthand such as "5m" or "2m" counts as money (millions) and is replaced, even though it could mean minutes or metres. This leans safe.
- **Number check:** every number in the draft must appear in the retrieved chunks. If one does not, the draft is rejected and a safe fallback is returned.
  - Digits inside words are checked too: "4pm" fails unless "4" appears in the retrieved chunks. This leans safe.
- **Web page notice:** "Use made-up messages only. Don't enter personal details."

## Engine

- One Gemini call per message.
- Default model `gemini-3.6-flash`, configurable via `CLIENT_AGENT_MODEL`.
- Uses its own API key from its own Google project (`techbygrace-inquiry-demo`), read from the `GEMINI_API_KEY` environment variable. Scripts never load `.env`.
- 60-second timeout.
- Retries on 503, 504 and dropped connections:
  - Command line: 2 retries, after 30 s and then 60 s.
  - Web: 1 retry, after 10 s.
- Never retry 429.

### Engine decisions (2026-11-09)

- **No reply cache.** Client messages are never stored; running the same message again makes a new Gemini call.
- **JSON output mode** is requested from Gemini (`response_mime_type="application/json"`).
- **Strict JSON validation.** An unknown category, or a missing or wrongly typed field (`category`, `reply`, `sources`, `needs_grace`), is an error. The agent never guesses or maps it to `other`.
- **The number check uses all retrieved chunks** (the top 5, plus any fallback chunks), not only the ones Gemini cited.
- **The fallback clears sources.** The safe fallback reply states no facts, so it lists no sources.

## Web

- FastAPI.
- Message limit: 2,000 characters.
- Rate limits: 5 per hour per visitor, 20 per day in total.
- `/health` answers both GET and HEAD.
- Port read from the `PORT` environment variable.
- Docker image, deployed on the Render free plan, monitored by UptimeRobot.

## Evaluation

`evaluation/` holds 10 made-up inquiries. Each one lists its expected category, the facts that must appear, and the things that must not appear. Required cases:

| Inquiry | Expected |
|---|---|
| Weekend question | Says no weekends |
| Refund question | Says no refunds |
| Code ownership | Client owns the code after final payment |
| Price question | No figures; the quote line is used |

Runner: `uv run python scripts/run_evaluation.py`. `--only E01,E05,E08,E10` runs some inquiries (ids or prefixes before the "-") and writes `results-<date>-only-<ids>.md`, so a partial run never overwrites the full results file (decision 2026-11-09).

Final result (2026-11-11): 10/10 inquiries passed every automatic check (first run 2026-11-09: 6/10); see `evaluation/results-2026-11-11.md`. The fixes were tuned on these same 10 inquiries, so a larger, unseen test set is future work.

## Known limitations (v1)

- The number check only covers numbers written in digits. Numbers written as words ("three weeks") are not checked.
- The 180-word limit on replies is an instruction to Gemini, not enforced in code.

## Out of scope for v1

- Sending email
- Conversation memory
- Booking calls
