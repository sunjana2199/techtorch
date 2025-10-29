# AI Engineer Technical Challenge – The Revenue Leakage Agent  

### Duration  
1.5 hours (core demo)

---

## Context
This challenge simulates an AI “financial detective” that investigates **revenue leakage** and can **propose and apply fixes** in a sandbox.  
All data is provided in JSON — no parsing required.

---

## Key Concepts
| Term | Meaning | Example Use |
|------|----------|--------------|
| **Credit Memo** | A negative invoice that reduces what the customer owes — used when overbilled or a pricing error occurred. | Overbilled €25 000 → $27 000 → issue credit memo for $2 000 USD |
| **Plan Amendment** | An update to the contract/billing plan (total, cadence, entitlements) when the agreement itself changes. | Upgrade plan from $90 000 → $100 000 or add “Premium Support” |
| **Make-Good Invoice** | A new invoice to recover missed or underbilled revenue. | Missing September billing → invoice $8 000 USD |

---

## Objective
Build an **AI agent** that can:

1. **Investigate** anomalies between billing plans and invoices  
2. **Propose** corrective actions (make-good invoice, credit memo, or plan amendment)  
3. **Apply** those actions to a writable **sandbox** with human approval  
4. **Explain** what it did and why, citing evidence or calculations  

The agent should **decide what to do**, not just execute a fixed pipeline.

---

## Data Overview (`/data/`)
| File | Description |
|------|--------------|
| `billing_plans.json` | Expected billing plans (contracts) |
| `invoices.json` | Issued invoices / bills |
| `credit_memos.json` | Existing credit memos |
| `exchange_rates.json` | FX rates |
| `/sandbox/*.json` | Writable ledgers for applied actions |

---

## Available Tools
| Tool | Purpose |
|------|----------|
| `load_plan(plan_id)` | Read plan details |
| `query_invoices(filters)` | Filter invoices by plan, customer, date |
| `fx_convert(amount, from_ccy, to_ccy, on_date)` | Currency conversion |
| `propose_make_good_invoice(plan_id, amount, reason)` | Draft new invoice |
| `propose_credit_memo(invoice_id, amount, reason)` | Draft credit memo |
| `propose_plan_amendment(plan_id, change_set)` | Draft billing plan update |
| `apply(action_draft)` | Write to `/sandbox` after confirmation |
| `rollback(action_id)` | Undo last applied action |

---

## UI Requirements
- Simple dashboard + chat (NextJS / React / Streamlit / ...)
- Input selector for mission & parameters
- Output tables: Findings, Proposals (with "Apply" buttons)
- Optional: reasoning trace & audit-log viewer (sandbox/audit_log.json)

---

## AI Tool Usage Tracking
**Important:** Document all AI tool prompts used during this challenge in `candidate_prompts.md`. This includes:
- ChatGPT, Claude, GitHub Copilot, Cursor, or any other AI tools
- Exact prompts used and their context
- Results and follow-up iterations
- Reflection on tool effectiveness

This documentation is part of the evaluation process and helps assess AI tool utilization skills.

---

## Submission
**Submit your solution via private GitHub repository:**

1. **Create a new private repository** on your personal GitHub account
2. **Copy the challenge files** to your new repository  
3. **Implement your solution** in the repository
4. **Add the reviewer** as a collaborator with read access
5. **Submit the repository URL** to the reviewer
