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

## Known limitations (v1)

- The number check only covers numbers written in digits. Numbers written as words ("three weeks") are not checked.

## Out of scope for v1

- Sending email
- Conversation memory
- Booking calls
