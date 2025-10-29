"""
Revenue Leakage Agent Tools
Implements all the tools mentioned in the readme for investigating and fixing revenue leakage.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class RevenueLeakageTools:
    def __init__(self, data_dir: str = "data", sandbox_dir: str = "sandbox"):
        self.data_dir = data_dir
        self.sandbox_dir = sandbox_dir
        self._load_data()
        load_dotenv()
        self._llm = None
    
    def _load_data(self):
        """Load all data files from the data directory"""
        with open(f"{self.data_dir}/billing_plans.json", "r") as f:
            self.billing_plans = json.load(f)
        
        with open(f"{self.data_dir}/invoices.json", "r") as f:
            self.invoices = json.load(f)
        
        with open(f"{self.data_dir}/credit_memos.json", "r") as f:
            self.credit_memos = json.load(f)
        
        with open(f"{self.data_dir}/exchange_rates.json", "r") as f:
            self.exchange_rates = json.load(f)
    
    def load_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Read plan details by plan_id"""
        for plan in self.billing_plans:
            if plan["plan_id"] == plan_id:
                return plan
        return None
    
    def query_invoices(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter invoices by plan, customer, date, etc."""
        filtered_invoices = []
        
        for invoice in self.invoices:
            match = True
            for key, value in filters.items():
                if key in invoice and invoice[key] != value:
                    match = False
                    break
            if match:
                filtered_invoices.append(invoice)
        
        return filtered_invoices
    
    def fx_convert(self, amount: float, from_ccy: str, to_ccy: str, on_date: str) -> float:
        """Convert currency using exchange rates"""
        if from_ccy == to_ccy:
            return amount
        
        # Find the exchange rate for the given date
        rate = None
        for fx_rate in self.exchange_rates:
            if (fx_rate["date"] == on_date and 
                fx_rate["from_currency"] == from_ccy and 
                fx_rate["to_currency"] == to_ccy):
                rate = fx_rate["rate"]
                break
        
        if rate is None:
            # If no specific rate found, try to find the closest available rate
            # Sort by date and find the most recent rate before the target date
            available_rates = [r for r in self.exchange_rates 
                             if r["from_currency"] == from_ccy and r["to_currency"] == to_ccy]
            
            if available_rates:
                # Sort by date (most recent first)
                available_rates.sort(key=lambda x: x["date"], reverse=True)
                # Use the most recent rate available
                rate = available_rates[0]["rate"]
            else:
                # If no rate found at all, return original amount with a warning
                print(f"Warning: No exchange rate found for {from_ccy} to {to_ccy} on {on_date}")
                return amount
        
        return amount * rate
    
    def propose_make_good_invoice(self, plan_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Draft a new make-good invoice"""
        plan = self.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        # Generate new invoice ID
        max_invoice_id = max([int(inv["invoice_id"].split("-")[1]) for inv in self.invoices])
        new_invoice_id = f"I-{max_invoice_id + 1}"
        
        proposal = {
            "action_type": "make_good_invoice",
            "proposal_id": f"PROP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "plan_id": plan_id,
            "customer_name": plan["customer_name"],
            "invoice_id": new_invoice_id,
            "amount": amount,
            "currency": plan["currency"],
            "reason": reason,
            "status": "draft",
            "created_at": datetime.now().isoformat()
        }
        
        return proposal
    
    def propose_credit_memo(self, invoice_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Draft a credit memo for an invoice"""
        # Find the invoice
        invoice = None
        for inv in self.invoices:
            if inv["invoice_id"] == invoice_id:
                invoice = inv
                break
        
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        
        # Generate new memo ID
        max_memo_id = max([int(memo["memo_id"].split("-")[1]) for memo in self.credit_memos])
        new_memo_id = f"M-{max_memo_id + 1}"
        
        proposal = {
            "action_type": "credit_memo",
            "proposal_id": f"PROP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "invoice_id": invoice_id,
            "plan_id": invoice["plan_id"],
            "customer_name": invoice["customer_name"],
            "memo_id": new_memo_id,
            "amount": amount,
            "currency": invoice["currency"],
            "reason": reason,
            "status": "draft",
            "created_at": datetime.now().isoformat()
        }
        
        return proposal
    
    def propose_plan_amendment(self, plan_id: str, change_set: Dict[str, Any]) -> Dict[str, Any]:
        """Draft a billing plan update"""
        plan = self.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        proposal = {
            "action_type": "plan_amendment",
            "proposal_id": f"PROP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "plan_id": plan_id,
            "customer_name": plan["customer_name"],
            "original_plan": plan,
            "changes": change_set,
            "status": "draft",
            "created_at": datetime.now().isoformat()
        }
        
        return proposal
    
    def apply(self, action_draft: Dict[str, Any]) -> Dict[str, Any]:
        """Write to sandbox after confirmation"""
        action_id = f"ACTION-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add action ID and timestamp
        action_draft["action_id"] = action_id
        action_draft["applied_at"] = datetime.now().isoformat()
        action_draft["status"] = "applied"
        
        # Load existing applied actions
        applied_actions_path = f"{self.sandbox_dir}/applied_actions.json"
        try:
            with open(applied_actions_path, "r") as f:
                applied_actions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            applied_actions = []
        
        # Add new action
        applied_actions.append(action_draft)
        
        # Save back to file
        with open(applied_actions_path, "w") as f:
            json.dump(applied_actions, f, indent=2)
        
        # Update audit log
        self._update_audit_log(action_id, "applied", action_draft)
        
        return {
            "action_id": action_id,
            "status": "applied",
            "message": f"Action {action_id} has been applied successfully"
        }
    
    def rollback(self, action_id: str) -> Dict[str, Any]:
        """Undo last applied action"""
        applied_actions_path = f"{self.sandbox_dir}/applied_actions.json"
        
        try:
            with open(applied_actions_path, "r") as f:
                applied_actions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No applied actions found"}
        
        # Find and remove the action
        action_found = False
        for i, action in enumerate(applied_actions):
            if action.get("action_id") == action_id:
                applied_actions.pop(i)
                action_found = True
                break
        
        if not action_found:
            return {"error": f"Action {action_id} not found"}
        
        # Save back to file
        with open(applied_actions_path, "w") as f:
            json.dump(applied_actions, f, indent=2)
        
        # Update audit log
        self._update_audit_log(action_id, "rollback", {"action_id": action_id})
        
        return {
            "action_id": action_id,
            "status": "rollback",
            "message": f"Action {action_id} has been rolled back"
        }
    
    def _update_audit_log(self, action_id: str, action_type: str, details: Dict[str, Any]):
        """Update the audit log with action details"""
        audit_log_path = f"{self.sandbox_dir}/audit_log.json"
        
        try:
            with open(audit_log_path, "r") as f:
                audit_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            audit_log = []
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "action_type": action_type,
            "details": details
        }
        
        audit_log.append(audit_entry)
        
        with open(audit_log_path, "w") as f:
            json.dump(audit_log, f, indent=2)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the complete audit log"""
        audit_log_path = f"{self.sandbox_dir}/audit_log.json"
        
        try:
            with open(audit_log_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_applied_actions(self) -> List[Dict[str, Any]]:
        """Get all applied actions"""
        applied_actions_path = f"{self.sandbox_dir}/applied_actions.json"
        
        try:
            with open(applied_actions_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def analyze_revenue_leakage(self) -> List[Dict[str, Any]]:
        """Analyze potential revenue leakage by comparing plans vs invoices"""
        findings = []
        
        # Get current date to only check up to current month
        from datetime import datetime
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Check for missing invoices
        for plan in self.billing_plans:
            plan_id = plan["plan_id"]
            plan_invoices = self.query_invoices({"plan_id": plan_id})
            
            if plan["cadence"] == "Monthly":
                expected_monthly = plan["total_value"] / 12
                # Check if we have invoices for each month up to current month
                for month in range(1, current_month + 1):  # Only check up to current month
                    month_invoices = [inv for inv in plan_invoices if inv["issue_date"].startswith(f"{current_year}-{month:02d}")]
                    if not month_invoices:
                        # Determine severity based on how recent the missing invoice is
                        months_behind = current_month - month
                        severity = "high" if months_behind <= 1 else "medium" if months_behind <= 3 else "low"
                        
                        findings.append({
                            "type": "missing_invoice",
                            "plan_id": plan_id,
                            "customer_name": plan["customer_name"],
                            "expected_amount": expected_monthly,
                            "currency": plan["currency"],
                            "month": f"{current_year}-{month:02d}",
                            "severity": severity
                        })
            
            elif plan["cadence"] == "Quarterly":
                expected_quarterly = plan["total_value"] / 4
                # Check quarterly invoices up to current quarter
                current_quarter = (current_month - 1) // 3 + 1
                quarters = ["Q1", "Q2", "Q3", "Q4"]
                for i, quarter in enumerate(quarters[:current_quarter]):  # Only check up to current quarter
                    quarter_month = (i + 1) * 3
                    quarter_invoices = [inv for inv in plan_invoices 
                                      if inv["issue_date"].startswith(f"{current_year}-{quarter_month:02d}")]
                    if not quarter_invoices:
                        # Determine severity based on how recent the missing invoice is
                        quarters_behind = current_quarter - (i + 1)
                        severity = "high" if quarters_behind <= 1 else "medium" if quarters_behind <= 2 else "low"
                        
                        findings.append({
                            "type": "missing_invoice",
                            "plan_id": plan_id,
                            "customer_name": plan["customer_name"],
                            "expected_amount": expected_quarterly,
                            "currency": plan["currency"],
                            "quarter": quarter,
                            "severity": severity
                        })
            
            elif plan["cadence"] == "Annual":
                # For annual plans, check if we're in the same year as the plan start date
                plan_start_year = int(plan["start_date"][:4])
                if current_year >= plan_start_year:
                    annual_invoices = [inv for inv in plan_invoices if inv["issue_date"].startswith(str(current_year))]
                    if not annual_invoices:
                        findings.append({
                            "type": "missing_invoice",
                            "plan_id": plan_id,
                            "customer_name": plan["customer_name"],
                            "expected_amount": plan["total_value"],
                            "currency": plan["currency"],
                            "year": str(current_year),
                            "severity": "high"
                        })
        
        # Check for overbilling (amounts higher than expected)
        for plan in self.billing_plans:
            plan_id = plan["plan_id"]
            plan_invoices = self.query_invoices({"plan_id": plan_id})
            
            if plan["cadence"] == "Monthly":
                expected_monthly = plan["total_value"] / 12
                for invoice in plan_invoices:
                    overage = invoice["amount_invoiced"] - expected_monthly
                    if overage > expected_monthly * 0.05:  # 5% tolerance for overbilling
                        # Determine severity based on overage percentage
                        overage_percentage = (overage / expected_monthly) * 100
                        if overage_percentage > 20:
                            severity = "high"
                        elif overage_percentage > 10:
                            severity = "medium"
                        else:
                            severity = "low"
                            
                        findings.append({
                            "type": "overbilling",
                            "plan_id": plan_id,
                            "customer_name": plan["customer_name"],
                            "invoice_id": invoice["invoice_id"],
                            "expected_amount": expected_monthly,
                            "actual_amount": invoice["amount_invoiced"],
                            "currency": plan["currency"],
                            "overage": overage,
                            "severity": severity
                        })
        
        # Check for currency conversion issues
        for invoice in self.invoices:
            if invoice["currency"] != "USD":
                # Check if there's a corresponding plan in USD
                plan = self.load_plan(invoice["plan_id"])
                if plan and plan["currency"] == "USD":
                    # Convert invoice amount to USD
                    converted_amount = self.fx_convert(
                        invoice["amount_invoiced"], 
                        invoice["currency"], 
                        "USD", 
                        invoice["issue_date"]
                    )
                    
                    # Calculate expected amount based on plan cadence
                    if plan["cadence"] == "Monthly":
                        expected_amount = plan["total_value"] / 12
                    elif plan["cadence"] == "Quarterly":
                        expected_amount = plan["total_value"] / 4
                    else:  # Annual
                        expected_amount = plan["total_value"]
                    
                    difference = abs(converted_amount - expected_amount)
                    if difference > expected_amount * 0.05:  # 5% tolerance
                        # Determine severity based on difference percentage
                        diff_percentage = (difference / expected_amount) * 100
                        if diff_percentage > 15:
                            severity = "high"
                        elif diff_percentage > 8:
                            severity = "medium"
                        else:
                            severity = "low"
                            
                        findings.append({
                            "type": "currency_conversion_issue",
                            "plan_id": invoice["plan_id"],
                            "customer_name": invoice["customer_name"],
                            "invoice_id": invoice["invoice_id"],
                            "original_amount": invoice["amount_invoiced"],
                            "original_currency": invoice["currency"],
                            "converted_amount": converted_amount,
                            "expected_amount": expected_amount,
                            "severity": severity
                        })
        
        return findings

    def _get_llm(self) -> Optional[ChatOpenAI]:
        """Lazily initialize and return an LLM client if configured."""
        if self._llm is not None:
            return self._llm
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            self._llm = ChatOpenAI(model="gpt-4", temperature=0.1, api_key=api_key)
            return self._llm
        except Exception:
            return None

    def analyze_revenue_leakage_llm(self) -> List[Dict[str, Any]]:
        """Use an LLM to analyze datasets and return structured leakage findings.

        Falls back to heuristic analysis if the LLM is unavailable or returns invalid output.
        """
        llm = self._get_llm()
        if llm is None:
            return self.analyze_revenue_leakage()

        # Build compact context for the model
        context = {
            "billing_plans": self.billing_plans,
            "invoices": self.invoices,
            "credit_memos": self.credit_memos,
            "exchange_rates": self.exchange_rates,
        }

        schema_description = (
            "Return ONLY a JSON array of findings. Each finding must be an object with: "
            "type one of ['missing_invoice','overbilling','currency_conversion_issue']; "
            "plan_id (str); customer_name (str); severity one of ['high','medium','low']. "
            "For 'missing_invoice': include expected_amount (number), currency (str), and one of month (YYYY-MM), quarter (Q1/Q2/Q3/Q4), or year (YYYY). "
            "For 'overbilling': include invoice_id (str), expected_amount (number), actual_amount (number), currency (str), overage (number). "
            "For 'currency_conversion_issue': include invoice_id (str), original_amount (number), original_currency (str), converted_amount (number), expected_amount (number)."
        )

        prompt = (
            "Analyze the provided billing plans, invoices, credit memos, and exchange rates to detect revenue leakage. "
            "Consider cadence vs expected billing, overbilling beyond reasonable tolerance (~5%), and FX conversion mismatches where plans are in USD but invoices are in another currency. "
            "Set severity higher for more recent or larger-impact discrepancies. "
            f"{schema_description}\n\n"
            "Data (JSON):\n" + json.dumps(context)
        )

        messages = [
            SystemMessage(content=(
                "You are a precise financial data analyst. You must return valid, parsable JSON only, "
                "matching the requested schema, without extra commentary."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = llm.invoke(messages)
            content = response.content or ""
            # Try to extract JSON array
            parsed: Optional[List[Dict[str, Any]]] = None
            text = content.strip()
            # If fenced code block, extract
            if text.startswith("```"):
                # Remove triple backticks and any language hint
                first_newline = text.find("\n")
                if first_newline != -1:
                    text = text[first_newline + 1 :]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            # If the model wrapped JSON in prose, attempt to find first [ ... ]
            if not text.startswith("["):
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    text = text[start : end + 1]
            parsed = json.loads(text)

            if not isinstance(parsed, list):
                raise ValueError("Model did not return a list")

            # Basic normalization/validation
            normalized: List[Dict[str, Any]] = []
            for f in parsed:
                if not isinstance(f, dict):
                    continue
                ftype = f.get("type")
                if ftype not in ["missing_invoice", "overbilling", "currency_conversion_issue"]:
                    continue
                # Ensure required shared fields
                if not f.get("plan_id") or not f.get("customer_name"):
                    continue
                sev = (f.get("severity") or "medium").lower()
                if sev not in ("high", "medium", "low"):
                    sev = "medium"
                f["severity"] = sev
                normalized.append(f)

            # If nothing valid, fallback
            if not normalized:
                return self.analyze_revenue_leakage()
            return normalized
        except Exception:
            return self.analyze_revenue_leakage()

    def answer_question(self, prompt: str) -> str:
        """Answer natural-language questions using the loaded data, powered by an LLM when available.

        Behavior:
        - If an OpenAI API key is configured, use the LLM with a compact JSON context of the datasets
          (optionally filtered to a mentioned customer) and direct it to answer concisely based on data.
        - If the LLM is unavailable or errors, fall back to deterministic heuristic summaries.
        """
        try:
            question = (prompt or "").strip()
            if not question:
                return "Please enter a question."

            ql = question.lower()

            # Identify customer mentioned in the prompt (simple substring match)
            known_customers = list({plan["customer_name"] for plan in self.billing_plans})
            mentioned_customer = None
            for name in known_customers:
                if name.lower() in ql:
                    mentioned_customer = name
                    break

            # Prepare filtered views for grounding
            billing_plans = self.billing_plans
            invoices = self.invoices
            credit_memos = self.credit_memos
            exchange_rates = self.exchange_rates
            if mentioned_customer:
                billing_plans = [p for p in billing_plans if p.get("customer_name") == mentioned_customer]
                invoices = [i for i in invoices if i.get("customer_name") == mentioned_customer]
                credit_memos = [m for m in credit_memos if m.get("customer_name") == mentioned_customer]

            # Prefer the LLM if available
            llm = self._get_llm()
            if llm is not None:
                try:
                    # Include both heuristic findings (for structure) and raw data context
                    findings = self.analyze_revenue_leakage_llm()
                    if mentioned_customer:
                        findings = [f for f in findings if f.get("customer_name") == mentioned_customer]

                    context = {
                        "billing_plans": billing_plans,
                        "invoices": invoices,
                        "credit_memos": credit_memos,
                        "exchange_rates": exchange_rates,
                        "derived_findings": findings,
                    }

                    system_msg = (
                        "You are a concise financial operations assistant. Answer based ONLY on the provided JSON data. "
                        "Cite numbers directly, avoid speculation, and keep answers brief (2-6 sentences)."
                    )

                    user_msg = (
                        "Question: "
                        + question
                        + "\n\nData (JSON):\n"
                        + json.dumps(context)
                    )

                    messages = [
                        SystemMessage(content=system_msg),
                        HumanMessage(content=user_msg),
                    ]

                    response = llm.invoke(messages)
                    if response and (response.content or "").strip():
                        return response.content.strip()
                except Exception:
                    # Fall through to heuristics
                    pass

            # Heuristic fallback below
            # Compute heuristic findings (unfiltered first, then filter for customer)
            findings_h = self.analyze_revenue_leakage()
            if mentioned_customer:
                findings_h = [f for f in findings_h if f.get("customer_name") == mentioned_customer]

            def summarize_leakage(fs: List[Dict[str, Any]]) -> str:
                if not fs:
                    if mentioned_customer:
                        return f"No revenue leakage detected for {mentioned_customer}."
                    return "No revenue leakage detected across customers."

                type_counts: Dict[str, int] = {}
                severity_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
                impact = 0.0
                for f in fs:
                    ftype = f.get("type", "unknown")
                    type_counts[ftype] = type_counts.get(ftype, 0) + 1
                    sev = f.get("severity")
                    if sev in severity_counts:
                        severity_counts[sev] += 1
                    if ftype == "missing_invoice":
                        impact += float(f.get("expected_amount", 0) or 0)
                    elif ftype == "overbilling":
                        impact += float(f.get("overage", 0) or 0)
                    elif ftype == "currency_conversion_issue":
                        impact += abs(float(f.get("converted_amount", 0) or 0) - float(f.get("expected_amount", 0) or 0))

                parts = []
                total_issues = len(fs)
                if mentioned_customer:
                    parts.append(f"Yes — revenue leakage detected for {mentioned_customer}: {total_issues} issues")
                else:
                    parts.append(f"Yes — revenue leakage detected: {total_issues} issues across customers")
                parts.append(
                    f"Severity: {severity_counts['high']} high, {severity_counts['medium']} medium, {severity_counts['low']} low"
                )
                parts.append(
                    "By type: " + ", ".join([f"{k.replace('_',' ')}: {v}" for k, v in sorted(type_counts.items())])
                )
                parts.append(f"Estimated impact: ${impact:,.0f}")
                return ". ".join(parts) + "."

            def summarize_billing() -> str:
                if not billing_plans and not invoices:
                    return "No billing data found."
                total_plans = len(billing_plans)
                total_value = sum(float(p.get("total_value", 0) or 0) for p in billing_plans)
                total_invoices = len(invoices)
                total_invoiced = sum(float(i.get("amount_invoiced", 0) or 0) for i in invoices)
                if mentioned_customer:
                    return (
                        f"Billing for {mentioned_customer}: {total_plans} plan(s), total plan value ${total_value:,.0f}; "
                        f"{total_invoices} invoice(s), total invoiced ${total_invoiced:,.0f}."
                    )
                return (
                    f"Billing summary: {total_plans} plans (${total_value:,.0f} total value); "
                    f"{total_invoices} invoices (${total_invoiced:,.0f} total invoiced)."
                )

            def summarize_currency(fs: List[Dict[str, Any]]) -> str:
                ccy_issues = [f for f in fs if f.get("type") == "currency_conversion_issue"]
                if not ccy_issues:
                    if mentioned_customer:
                        return f"No currency conversion issues detected for {mentioned_customer}."
                    return "No currency conversion issues detected."
                impact = sum(
                    abs(float(f.get("converted_amount", 0) or 0) - float(f.get("expected_amount", 0) or 0)) for f in ccy_issues
                )
                return f"Currency conversion issues: {len(ccy_issues)} issue(s), est. impact ${impact:,.0f}."

            asks_leakage = any(k in ql for k in ["leakage", "missing", "overbill", "over-bill", "currency conversion", "fx"])
            asks_billing = any(k in ql for k in ["invoice", "invoices", "billing", "bill", "plan", "plans"])
            asks_currency = any(k in ql for k in ["currency", "fx", "conversion", "exchange rate"])

            responses: List[str] = []
            if asks_leakage:
                responses.append(summarize_leakage(findings_h))
            if asks_currency and not asks_leakage:
                responses.append(summarize_currency(findings_h))
            if asks_billing and not asks_leakage:
                responses.append(summarize_billing())

            if responses:
                return "\n".join(responses)

            overview = summarize_billing()
            leakage_line = summarize_leakage(findings_h)
            if leakage_line.startswith("No revenue leakage"):
                return overview + "\n" + leakage_line
            else:
                return leakage_line + "\n" + overview

        except Exception as exc:
            return f"Sorry, I couldn't answer that due to an error: {exc}"
