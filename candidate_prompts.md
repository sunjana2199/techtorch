# Candidate AI Tool Usage Log

**Instructions for Candidates:**
Please document all prompts you use with AI tools during this challenge. This helps us understand your problem-solving approach and AI tool utilization skills.

---

## AI Tool Usage Log

### Prompt 1
**Tool Used:** Cursor + ChatGPT
**Context:** Scaffold a Streamlit billing dashboard with KPIs, filters, charts, and currency conversion.
**Prompt:**
```
Build a Streamlit dashboard that reads local JSON files (data/invoices.json, data/billing_plans.json, data/credit_memos.json, data/exchange_rates.json). Show a high-level billing overview with KPI cards (Total Invoices, Total Due, Paid %, MRR), a filterable invoices table (by customer, status, and date range), and basic charts. Convert all monetary amounts to a selected currency using exchange_rates.json (default USD). Provide clean, typed helper functions and avoid global state.
```

**Result:** Generated an initial `app.py` layout with KPI cards, filters, table, and currency selector, plus helper stubs for conversions and caching hooks.
**Follow-up:** Asked for robust currency conversion and date filtering; added fallback handling when JSON files are missing and improved default states.

---

### Prompt 2
**Tool Used:** ChatGPT (inside Cursor)
**Context:** Design `tools.py` utilities for loading/normalizing JSON data, computing metrics (MRR, churn, ARPA), currency conversion, overdue detection, and credit memo application.
**Prompt:**
```
Create a Python utilities module (tools.py) with functions to: load and normalize invoice, plan, credit memo, and exchange rate data; compute MRR, churn rate, ARPA; convert amounts between currencies using a base currency; detect overdue invoices; and apply credit memos to invoices with correct remaining balances. Return pandas DataFrames where convenient and plain dicts elsewhere. Include unit-testable pure functions.
```

**Result:** Produced a cohesive utilities module with pure, testable functions (minimal side effects), optional pandas interop, and clear error handling.
**Follow-up:** Iterated to improve precision/rounding, clarified return types, and added guards for missing exchange rates.

---

### Prompt 3
**Tool Used:** Cursor
**Context:** Add sidebar actions to apply credits, mark invoices paid, and write off amounts; persist to `sandbox/applied_actions.json` and append to `sandbox/audit_log.json` with timestamps.
**Prompt:**
```
Add an "Actions" or sidebar workflow in the Streamlit app to apply adjustments (e.g., apply a credit memo to an invoice, mark invoice as paid, write-offs). Persist each action to sandbox/applied_actions.json and append an audit entry to sandbox/audit_log.json with timestamp, user, action type, inputs, and results. Make actions idempotent and re-runnable across app sessions.
```

**Result:** Implemented action handlers with validation, idempotency checks, persistence, and audit logging; exposed a simple UI to trigger and review.
**Follow-up:** Refined schema (action name, params, result), added collision handling, and improved user feedback messages.

---

### Prompt 4
**Tool Used:** ChatGPT
**Context:** Create a lightweight agent in `agent.py` to parse natural language ("show overdue invoices for October", "apply $100 credit to INV-123 in EUR") and route to tool functions.
**Prompt:**
```
Implement a lightweight agent in agent.py that takes natural language commands (e.g., "show overdue invoices for October" or "apply $100 credit to INV-123 in EUR") and routes them to tool functions. Define a small action schema (name, params, handler) and robust parameter parsing/validation. Return structured results for the UI to render.
```

**Result:** Added an intent/parameter parser, action registry (name → handler), and structured responses for the UI; included validation and error surfaces.
**Follow-up:** Tuned parsing heuristics, expanded supported synonyms, and added safer default behaviors when parameters are ambiguous.

---

### Prompt 5
**Tool Used:** Cursor
**Context:** Polish UI/UX, add README and `requirements.txt`, ensure caching and empty-state handling, and improve performance.
**Prompt:**
```
Polish the UI/UX: KPI cards at top, date range picker, currency selector, customer multi-select, and a download CSV button for filtered invoices. Add a concise README with setup steps, `requirements.txt`, and a `streamlit run app.py` instruction. Ensure no secrets, handle missing files gracefully, and keep the app snappy with caching.
```

**Result:** Delivered KPI cards layout, date range picker, currency and customer filters, CSV export, and docs with setup/run steps.
**Follow-up:** Requested performance tweaks (`st.cache_data`) and added graceful handling for missing/empty datasets and invalid filter states.

---

**Note:** Add more prompt sections as needed. The goal is to capture your complete AI tool usage pattern during this challenge.
