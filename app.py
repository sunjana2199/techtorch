"""
Revenue Leakage Agent - Simplified Dashboard
A streamlined interface for detecting and fixing revenue leakage.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from agent import RevenueLeakageAgent
from tools import RevenueLeakageTools
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Revenue Leakage Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = RevenueLeakageAgent()
if "leakage_findings" not in st.session_state:
    st.session_state.leakage_findings = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_row" not in st.session_state:
    st.session_state.selected_row = None

def _finding_matches_action(finding, action):
    """Return True if the applied action resolves the given finding."""
    try:
        action_type = action.get("action_type")
        if finding.get("type") in ("overbilling", "currency_conversion_issue"):
            # Resolved by a credit memo on the same invoice
            return action_type == "credit_memo" and action.get("invoice_id") == finding.get("invoice_id")
        if finding.get("type") == "missing_invoice":
            # Resolved by a make-good invoice for the same plan and same period mentioned in reason
            if action_type != "make_good_invoice":
                return False
            if action.get("plan_id") != finding.get("plan_id"):
                return False
            reason = (action.get("reason") or "").lower()
            # Period might be in month, quarter, or year
            period = finding.get("month") or finding.get("quarter") or finding.get("year")
            if not period:
                return False
            return str(period).lower() in reason
    except Exception:
        return False
    return False

def _filter_out_applied(findings, applied_actions):
    """Remove findings that have already been resolved by applied actions."""
    if not applied_actions:
        return findings
    remaining = []
    for f in findings:
        matched = any(_finding_matches_action(f, a) for a in applied_actions)
        if not matched:
            remaining.append(f)
    return remaining

def main():
    st.title("💰 Revenue Leakage Dashboard")
    st.markdown("Simple interface to detect and fix revenue leakage issues")
    
    # Sidebar for mission selection
    with st.sidebar:
        st.header("Mission Control")
        
        # Mission selector
        mission_options = [
            "Detect revenue leakage",
            "Analyze specific customer",
            "Check billing discrepancies",
            "Review currency conversions"
        ]
        
        selected_mission = st.selectbox("Select Mission", mission_options)
        
        # Customer filter
        customers = ["All"] + list(set([plan["customer_name"] for plan in st.session_state.agent.tools.billing_plans]))
        customer_filter = st.selectbox("Customer Filter", customers)
        
        # Run analysis button
        if st.button("🔍 Analyze Revenue", type="primary"):
            with st.spinner("Analyzing revenue data..."):
                try:
                    # Always use LLM-powered analysis (with internal fallback to heuristics)
                    st.session_state.leakage_findings = st.session_state.agent.tools.analyze_revenue_leakage_llm()
                    st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        # Chat interface in sidebar
        st.header("💬 Chat")
        display_chat_interface()
    
    # Main content area with tabs
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Billing Plans", "🧾 Invoices", "🔍 Leakage Detection", "📜 History"])
    
    with tab1:
        display_billing_plans(customer_filter)
    
    with tab2:
        display_invoices(customer_filter)
    
    with tab3:
        display_leakage_detection()

    with tab4:
        display_history()
    
    # Popup for row details
    if st.session_state.selected_row is not None:
        display_row_popup()

def display_billing_plans(customer_filter):
    """Display billing plans in a clean table"""
    st.header("📊 Billing Plans")
    
    plans = st.session_state.agent.tools.billing_plans
    if customer_filter != "All":
        plans = [plan for plan in plans if plan["customer_name"] == customer_filter]
    
    if plans:
        plans_df = pd.DataFrame(plans)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Plans", len(plans))
        with col2:
            total_value = sum(plan["total_value"] for plan in plans)
            st.metric("Total Value", f"${total_value:,.0f}")
        with col3:
            currencies = set(plan["currency"] for plan in plans)
            st.metric("Currencies", len(currencies))
        
        # Display plans table
        st.dataframe(plans_df, use_container_width=True)
    else:
        st.info("No billing plans found for the selected customer.")

def display_invoices(customer_filter):
    """Display invoices in a clean table"""
    st.header("🧾 Invoices")
    
    invoices = st.session_state.agent.tools.invoices
    if customer_filter != "All":
        invoices = [invoice for invoice in invoices if invoice["customer_name"] == customer_filter]
    
    if invoices:
        invoices_df = pd.DataFrame(invoices)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Invoices", len(invoices))
        with col2:
            total_amount = sum(invoice["amount_invoiced"] for invoice in invoices)
            st.metric("Total Amount", f"${total_amount:,.0f}")
        with col3:
            paid_count = len([inv for inv in invoices if inv["status"] == "paid"])
            st.metric("Paid Invoices", f"{paid_count}/{len(invoices)}")
        
        # Display invoices table
        st.dataframe(invoices_df, use_container_width=True)
    else:
        st.info("No invoices found for the selected customer.")

def display_leakage_detection():
    """Display leakage detection results with filterable tags and apply buttons"""
    st.header("🔍 Revenue Leakage Detection")
    
    if not st.session_state.leakage_findings:
        st.info("Click 'Analyze Revenue' in the sidebar to detect leakage issues.")
        return
    
    findings = st.session_state.leakage_findings
    # Exclude findings with already applied fixes
    applied_actions = st.session_state.agent.get_applied_actions()
    findings = _filter_out_applied(findings, applied_actions)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_count = len([f for f in findings if f.get("severity") == "high"])
        st.metric("🔴 High Severity", high_count, delta=None)
    with col2:
        medium_count = len([f for f in findings if f.get("severity") == "medium"])
        st.metric("🟡 Medium Severity", medium_count, delta=None)
    with col3:
        low_count = len([f for f in findings if f.get("severity") == "low"])
        st.metric("🟢 Low Severity", low_count, delta=None)
    with col4:
        total_value = sum(f.get("expected_amount", 0) for f in findings if f["type"] == "missing_invoice")
        total_value += sum(f.get("overage", 0) for f in findings if f["type"] == "overbilling")
        total_value += sum(abs(f.get("converted_amount", 0) - f.get("expected_amount", 0)) for f in findings if f["type"] == "currency_conversion_issue")
        st.metric("💰 Total Impact", f"${total_value:,.0f}", delta=None)
    
    # Filter options
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    
    with col2:
        type_filter = st.multiselect(
            "Filter by Type",
            list(set(finding["type"] for finding in findings)),
            default=list(set(finding["type"] for finding in findings))
        )
    
    # Filter findings
    filtered_findings = [
        f for f in findings 
        if f.get("severity") in severity_filter and f["type"] in type_filter
    ]
    
    if not filtered_findings:
        st.warning("No findings match the selected filters.")
        return
    
    # Display findings table
    st.subheader("Leakage Issues Found")
    
    for i, finding in enumerate(filtered_findings):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                severity_color = "🔴" if finding.get("severity") == "high" else "🟡" if finding.get("severity") == "medium" else "🟢"
                st.write(f"{severity_color} **{finding.get('type', 'unknown').replace('_', ' ').title()}**")
                st.write(f"Customer: {finding.get('customer_name', 'Unknown')}")
            
            with col2:
                if finding["type"] == "missing_invoice":
                    st.write(f"Expected: ${finding.get('expected_amount', 0):,.2f}")
                elif finding["type"] == "overbilling":
                    st.write(f"Overage: ${finding.get('overage', 0):,.2f}")
                elif finding["type"] == "currency_conversion_issue":
                    converted = finding.get('converted_amount', 0)
                    expected = finding.get('expected_amount', 0)
                    st.write(f"Difference: ${converted - expected:,.2f}")
            
            with col3:
                st.write(f"Currency: {finding.get('currency', 'N/A')}")
                if finding["type"] == "missing_invoice":
                    period = finding.get('month', finding.get('quarter', finding.get('year', 'period')))
                    st.write(f"Period: {period}")
                elif finding["type"] == "overbilling":
                    st.write(f"Invoice: {finding.get('invoice_id', 'N/A')}")
                elif finding["type"] == "currency_conversion_issue":
                    st.write(f"Invoice: {finding.get('invoice_id', 'N/A')}")
            
            with col4:
                if st.button("View Details", key=f"view_{i}"):
                    st.session_state.selected_row = finding
                    st.rerun()
                
                if st.button("Apply Fix", key=f"apply_{i}"):
                    with st.spinner("Applying fix..."):
                        apply_fix(finding)
                        st.rerun()
    
    # Display applied actions if any
    if applied_actions:
        st.subheader("✅ Applied Actions")
        for action in applied_actions:
            with st.expander(f"Action: {action.get('action_type', 'Unknown').replace('_', ' ').title()}"):
                st.json(action)

def display_chat_interface():
    """Display the simplified chat interface"""
    # Chat history
    if st.session_state.chat_history:
        st.write("**Recent Messages:**")
        for message in st.session_state.chat_history[-3:]:  # Show last 3 messages
            if message["role"] == "user":
                st.write(f"👤 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Agent:** {message['content']}")
        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about revenue leakage..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get agent response
        with st.spinner("Thinking..."):
            try:
                # Data-driven response based on loaded datasets
                response = st.session_state.agent.tools.answer_question(prompt)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()

def display_row_popup():
    """Display popup with detailed information about selected row"""
    if st.session_state.selected_row is None:
        return
    
    finding = st.session_state.selected_row
    
    # Create modal-like popup using columns
    st.markdown("---")
    st.subheader("🔍 Issue Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Type:** {finding['type'].replace('_', ' ').title()}")
        st.write(f"**Customer:** {finding['customer_name']}")
        st.write(f"**Severity:** {finding['severity'].upper()}")
        
        # Summary of finding
        st.subheader("Summary of Finding")
        try:
            if finding["type"] == "missing_invoice":
                period = finding.get('month', finding.get('quarter', finding.get('year', 'period')))
                st.write(
                    f"Missing invoice detected for {period}. Expected billing of "
                    f"${finding['expected_amount']:,.2f} {finding['currency']} was not issued."
                )
            elif finding["type"] == "overbilling":
                overage = finding.get('overage', 0)
                st.write(
                    f"Overbilling detected on invoice {finding.get('invoice_id', 'N/A')}: "
                    f"charged ${finding.get('actual_amount', 0):,.2f} vs expected ${finding.get('expected_amount', 0):,.2f} "
                    f"({overage:,.2f} {finding.get('currency','')} over)."
                )
            elif finding["type"] == "currency_conversion_issue":
                diff = finding.get('converted_amount', 0) - finding.get('expected_amount', 0)
                st.write(
                    f"Currency conversion discrepancy on invoice {finding.get('invoice_id','N/A')}: "
                    f"converted ${finding.get('converted_amount', 0):,.2f} USD vs expected ${finding.get('expected_amount', 0):,.2f} USD "
                    f"(difference {diff:,.2f} USD)."
                )
        except Exception:
            pass
        
        # Show evidence and calculations
        st.subheader("Evidence & Calculations")
        
        if finding["type"] == "missing_invoice":
            st.write(f"**Expected Amount:** ${finding['expected_amount']:,.2f} {finding['currency']}")
            st.write(f"**Period:** {finding.get('month', finding.get('quarter', finding.get('year', 'period')))}")
            st.write(f"**Plan ID:** {finding['plan_id']}")
            
        elif finding["type"] == "overbilling":
            st.write(f"**Expected Amount:** ${finding['expected_amount']:,.2f} {finding['currency']}")
            st.write(f"**Actual Amount:** ${finding['actual_amount']:,.2f} {finding['currency']}")
            st.write(f"**Overage:** ${finding['overage']:,.2f} {finding['currency']}")
            st.write(f"**Invoice ID:** {finding['invoice_id']}")
            
        elif finding["type"] == "currency_conversion_issue":
            st.write(f"**Original Amount:** {finding['original_amount']:,.2f} {finding['original_currency']}")
            st.write(f"**Converted Amount:** ${finding['converted_amount']:,.2f} USD")
            st.write(f"**Expected Amount:** ${finding['expected_amount']:,.2f} USD")
            st.write(f"**Difference:** ${finding['converted_amount'] - finding['expected_amount']:,.2f} USD")
            st.write(f"**Invoice ID:** {finding['invoice_id']}")
    
    with col2:
        # Proposed action
        st.subheader("Proposed Action")
        
        if finding["type"] == "missing_invoice":
            st.write("**Action:** Create Make-Good Invoice")
            st.write(f"**Amount:** ${finding['expected_amount']:,.2f} {finding['currency']}")
            st.write(f"**Reason:** Missing {finding.get('month', finding.get('quarter', finding.get('year', 'period')))} billing")
            
        elif finding["type"] == "overbilling":
            st.write("**Action:** Create Credit Memo")
            st.write(f"**Amount:** ${finding['overage']:,.2f} {finding['currency']}")
            st.write(f"**Reason:** Overbilling adjustment")
            
        elif finding["type"] == "currency_conversion_issue":
            overage = finding['converted_amount'] - finding['expected_amount']
            if overage > 0:
                st.write("**Action:** Create Credit Memo")
                st.write(f"**Amount:** ${overage:,.2f} USD")
                st.write(f"**Reason:** Currency conversion overbilling")
        
        # Action buttons
        if st.button("✅ Apply Fix", type="primary"):
            apply_fix(finding)
        
        if st.button("❌ Cancel"):
            st.session_state.selected_row = None
            st.rerun()
    
    st.markdown("---")

def apply_fix(finding):
    """Apply the proposed fix for a finding"""
    try:
        if finding["type"] == "missing_invoice":
            proposal = st.session_state.agent.tools.propose_make_good_invoice(
                plan_id=finding["plan_id"],
                amount=finding["expected_amount"],
                reason=f"Missing {finding.get('month', finding.get('quarter', finding.get('year', 'period')))} billing for {finding['customer_name']}"
            )
        elif finding["type"] == "overbilling":
            proposal = st.session_state.agent.tools.propose_credit_memo(
                invoice_id=finding["invoice_id"],
                amount=finding["overage"],
                reason=f"Overbilling adjustment for {finding['customer_name']} - {finding['overage']:.2f} {finding['currency']} over expected amount"
            )
        elif finding["type"] == "currency_conversion_issue":
            overage = finding["converted_amount"] - finding["expected_amount"]
            if overage > 0:
                proposal = st.session_state.agent.tools.propose_credit_memo(
                    invoice_id=finding["invoice_id"],
                    amount=overage,
                    reason=f"Currency conversion overbilling for {finding['customer_name']} - {overage:.2f} USD over expected amount"
                )
            else:
                st.error("No overbilling detected in currency conversion")
                return
        else:
            st.error(f"Unknown finding type: {finding['type']}")
            return
        
        # Apply the proposal
        result = st.session_state.agent.tools.apply(proposal)
        
        st.success(f"✅ Fix applied successfully! Action ID: {result['action_id']}")
        
        # Clear selected row and remove this finding from the dashboard immediately
        st.session_state.selected_row = None
        def _same_finding(a, b):
            if a.get("type") != b.get("type"):
                return False
            if a.get("type") in ("overbilling", "currency_conversion_issue"):
                return a.get("invoice_id") == b.get("invoice_id")
            if a.get("type") == "missing_invoice":
                return (
                    a.get("plan_id") == b.get("plan_id") and
                    (a.get("month") == b.get("month") or a.get("quarter") == b.get("quarter") or a.get("year") == b.get("year"))
                )
            return False
        st.session_state.leakage_findings = [f for f in st.session_state.leakage_findings if not _same_finding(f, finding)]
        
    except Exception as e:
        st.error(f"❌ Failed to apply fix: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

def display_history():
    """Display applied actions and audit log with rollback option"""
    st.header("📜 History")

    # Applied actions with rollback controls
    applied_actions = st.session_state.agent.get_applied_actions()
    if applied_actions:
        st.subheader("Applied Actions")
        for idx, action in enumerate(reversed(applied_actions)):
            action_id = action.get("action_id", f"unknown-{idx}")
            title = action.get("action_type", "Unknown").replace("_", " ").title()
            with st.expander(f"{title} — {action_id}"):
                col_a, col_b = st.columns([4,1])
                with col_a:
                    st.json(action)
                with col_b:
                    if st.button("Rollback", key=f"rollback_{action_id}"):
                        with st.spinner("Rolling back action..."):
                            result = st.session_state.agent.rollback_action(action_id)
                            if result.get("status") == "rollback" and not result.get("error"):
                                st.success(f"Rolled back {action_id}")
                                # refresh findings and rerun to update UI
                                st.session_state.leakage_findings = st.session_state.agent.tools.analyze_revenue_leakage()
                                st.rerun()
                            else:
                                st.error(result.get("error", "Failed to rollback"))
    else:
        st.info("No applied actions found.")

    st.markdown("---")

    # Audit log
    audit_log = st.session_state.agent.get_audit_log()
    st.subheader("Audit Log")
    if audit_log:
        # Build a quick lookup for currently applied actions
        applied_ids = {a.get("action_id") for a in applied_actions}
        # Show most recent first
        for entry in reversed(audit_log):
            action_id = entry.get('action_id','')
            title = f"{entry.get('timestamp', '')} — {entry.get('action_type','').title()} — {action_id}"
            with st.expander(title):
                col1, col2 = st.columns([4,1])
                with col1:
                    st.json(entry)
                with col2:
                    # Offer rollback here if this entry represents an applied action that is still applied
                    if entry.get('action_type') == 'applied' and action_id in applied_ids:
                        if st.button("Rollback", key=f"rollback_log_{action_id}"):
                            with st.spinner("Rolling back action..."):
                                result = st.session_state.agent.rollback_action(action_id)
                                if result.get("status") == "rollback" and not result.get("error"):
                                    st.success(f"Rolled back {action_id}")
                                    st.session_state.leakage_findings = st.session_state.agent.tools.analyze_revenue_leakage()
                                    st.rerun()
                                else:
                                    st.error(result.get("error", "Failed to rollback"))
    else:
        st.info("Audit log is empty.")

if __name__ == "__main__":
    main()
