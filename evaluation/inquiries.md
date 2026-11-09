# Evaluation inquiries

Ten made-up client messages (no real people or businesses). `scripts/run_evaluation.py` reads this file.

Format, one `##` section per inquiry:

- `message`: the client's message, on one line.
- `category`: the expected category. `a | b` means either is accepted.
- `must_include`: items separated by `;`. Each item must appear in the reply (case-insensitive). `a | b` inside an item means any one of them is enough.
- `must_not_include`: items separated by `;`. None may appear in the reply (case-insensitive).
- `needs_grace`: `yes` (must list at least one question for Grace), `no` (must be empty) or `any`.
- `-` means none.

Every reply is also checked automatically for money amounts (`find_money_amounts` must find nothing).

## E01-weekend
- message: Hi Grace, I'm busy with my shop during the week. Could we have our project calls on Saturdays or Sundays instead?
- category: policy
- must_include: Monday to Friday | weekdays | working days
- must_not_include: Saturday works; Sunday works; happy to meet on Saturday; happy to meet on Sunday; I can do Saturday; I can do Sunday
- needs_grace: any

## E02-refund
- message: If I change my mind halfway through the project, can I get a refund of my deposit?
- category: policy
- must_include: non-refundable | not refundable | no refunds | don't offer refunds | do not offer refunds | can't offer a refund | cannot offer a refund; keep everything | keep all the work | keep the work | keep what
- must_not_include: full refund; refund your deposit; money-back; money back guarantee
- needs_grace: any

## E03-code-ownership
- message: Once the app is finished, who owns the code — me or you?
- category: policy
- must_include: own; final payment
- must_not_include: I keep the code; I retain; licence fee; license fee
- needs_grace: any

## E04-price
- message: How much do you charge for a five-page business website?
- category: pricing
- must_include: Pricing depends on scope; quote after a short call
- must_not_include: ₦; $; NGN; USD
- needs_grace: any

## E05-new-project
- message: Hello, I'd like a website for my tailoring business with a booking form so customers can schedule fittings. How do we get started?
- category: new_project
- must_include: call; proposal
- must_not_include: ₦; $; NGN; USD
- needs_grace: any

## E06-timeline
- message: Roughly how long does it take to build a web app with user accounts and online payments?
- category: timeline
- must_include: 4 to 8 weeks | 4-8 weeks | 4–8 weeks
- must_not_include: next week; ₦; $
- needs_grace: any

## E07-support-after-delivery
- message: My website went live recently. If something breaks after launch, will you still fix it?
- category: support
- must_include: 1 to 2 months | 1-2 months | 1–2 months; support
- must_not_include: ₦; $
- needs_grace: any

## E08-unanswerable
- message: I have an idea for a mobile game for iPhone and Android. Do you build mobile games?
- category: new_project | other
- must_include: -
- must_not_include: yes, I build mobile games; I can build your game; I build mobile games; I'd love to build your game
- needs_grace: yes

## E09-mixed-topics
- message: Two quick questions: do you work on weekends, and how long does a simple business website usually take?
- category: policy | timeline
- must_include: Monday to Friday | weekdays | working days; 1 to 2 weeks | 1-2 weeks | 1–2 weeks
- must_not_include: ₦; $
- needs_grace: any

## E10-vague
- message: hi, interested
- category: other | new_project
- must_include: call | more about | tell me | details
- must_not_include: ₦; $
- needs_grace: any
