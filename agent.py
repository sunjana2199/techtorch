"""
Revenue Leakage Agent using LangGraph
Implements the AI agent that investigates, proposes, and applies fixes for revenue leakage.
"""

import json
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from tools import RevenueLeakageTools
import os
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    """State for the Revenue Leakage Agent"""
    mission: str
    findings: List[Dict[str, Any]]
    proposals: List[Dict[str, Any]]
    applied_actions: List[Dict[str, Any]]
    reasoning: str
    current_step: str
    user_input: str


class RevenueLeakageAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = RevenueLeakageTools()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("investigate", self.investigate_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("propose", self.propose_node)
        workflow.add_node("apply", self.apply_node)
        workflow.add_node("explain", self.explain_node)
        
        # Add edges
        workflow.set_entry_point("investigate")
        workflow.add_edge("investigate", "analyze")
        workflow.add_edge("analyze", "propose")
        workflow.add_edge("propose", "apply")
        workflow.add_edge("apply", "explain")
        workflow.add_edge("explain", END)
        
        return workflow.compile()
    
    def investigate_node(self, state: AgentState) -> AgentState:
        """Investigate revenue leakage by analyzing data"""
        state["current_step"] = "investigating"
        
        # Run automated analysis
        findings = self.tools.analyze_revenue_leakage()
        state["findings"] = findings
        
        # Use LLM to provide additional insights
        investigation_prompt = f"""
        As a financial detective, investigate potential revenue leakage based on these findings:
        
        {json.dumps(findings, indent=2)}
        
        Please provide additional analysis and insights about:
        1. Patterns in the data that might indicate systematic issues
        2. Priority levels for each finding
        3. Potential root causes
        4. Recommended next steps
        
        Be specific and cite evidence from the data.
        """
        
        messages = [
            SystemMessage(content="You are a financial detective specializing in revenue leakage detection. Analyze the provided findings and provide detailed insights."),
            HumanMessage(content=investigation_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["reasoning"] = response.content
        
        return state
    
    def analyze_node(self, state: AgentState) -> AgentState:
        """Analyze findings and determine what actions to take"""
        state["current_step"] = "analyzing"
        
        analysis_prompt = f"""
        Based on the investigation findings and reasoning, analyze what corrective actions should be taken:
        
        Findings: {json.dumps(state["findings"], indent=2)}
        Reasoning: {state["reasoning"]}
        
        For each finding, determine:
        1. What type of corrective action is needed (make-good invoice, credit memo, plan amendment)
        2. The specific parameters for each action
        3. The priority and urgency
        4. Any dependencies between actions
        
        Provide a structured analysis with clear recommendations.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst. Analyze the findings and determine the appropriate corrective actions."),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["reasoning"] += f"\n\nAnalysis:\n{response.content}"
        
        return state
    
    def propose_node(self, state: AgentState) -> AgentState:
        """Generate specific proposals for corrective actions"""
        state["current_step"] = "proposing"
        proposals = []
        
        for finding in state["findings"]:
            if finding["type"] == "missing_invoice":
                # Propose make-good invoice
                try:
                    proposal = self.tools.propose_make_good_invoice(
                        plan_id=finding["plan_id"],
                        amount=finding["expected_amount"],
                        reason=f"Missing {finding.get('month', finding.get('quarter', finding.get('year', 'period')))} billing for {finding['customer_name']}"
                    )
                    proposals.append(proposal)
                except Exception as e:
                    print(f"Error creating make-good invoice proposal: {e}")
            
            elif finding["type"] == "overbilling":
                # Propose credit memo
                try:
                    proposal = self.tools.propose_credit_memo(
                        invoice_id=finding["invoice_id"],
                        amount=finding["overage"],
                        reason=f"Overbilling adjustment for {finding['customer_name']} - {finding['overage']:.2f} {finding['currency']} over expected amount"
                    )
                    proposals.append(proposal)
                except Exception as e:
                    print(f"Error creating credit memo proposal: {e}")
            
            elif finding["type"] == "currency_conversion_issue":
                # Propose credit memo for currency conversion error
                try:
                    overage = finding["converted_amount"] - finding["expected_amount"]
                    if overage > 0:
                        proposal = self.tools.propose_credit_memo(
                            invoice_id=finding["invoice_id"],
                            amount=overage,
                            reason=f"Currency conversion overbilling for {finding['customer_name']} - {overage:.2f} USD over expected amount"
                        )
                        proposals.append(proposal)
                except Exception as e:
                    print(f"Error creating currency conversion credit memo proposal: {e}")
        
        state["proposals"] = proposals
        
        # Use LLM to review and enhance proposals
        review_prompt = f"""
        Review these proposed corrective actions:
        
        {json.dumps(proposals, indent=2)}
        
        Please:
        1. Validate each proposal for accuracy
        2. Suggest any improvements or additional considerations
        3. Prioritize the proposals by urgency and impact
        4. Identify any potential conflicts or dependencies
        
        Provide a summary of recommendations.
        """
        
        messages = [
            SystemMessage(content="You are a financial operations manager. Review and validate the proposed corrective actions."),
            HumanMessage(content=review_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["reasoning"] += f"\n\nProposal Review:\n{response.content}"
        
        return state
    
    def apply_node(self, state: AgentState) -> AgentState:
        """Apply the proposed actions (simulated - requires user approval)"""
        state["current_step"] = "applying"
        applied_actions = []
        
        for proposal in state["proposals"]:
            try:
                # In a real implementation, this would require user approval
                # For now, we'll simulate the application
                result = self.tools.apply(proposal)
                applied_actions.append({
                    "proposal": proposal,
                    "result": result,
                    "status": "applied"
                })
            except Exception as e:
                applied_actions.append({
                    "proposal": proposal,
                    "result": {"error": str(e)},
                    "status": "failed"
                })
        
        state["applied_actions"] = applied_actions
        
        return state
    
    def explain_node(self, state: AgentState) -> AgentState:
        """Explain what was done and why"""
        state["current_step"] = "explaining"
        
        explanation_prompt = f"""
        Provide a comprehensive explanation of the revenue leakage investigation and corrective actions:
        
        Mission: {state.get('mission', 'Investigate revenue leakage')}
        Findings: {json.dumps(state['findings'], indent=2)}
        Proposals: {json.dumps(state['proposals'], indent=2)}
        Applied Actions: {json.dumps(state['applied_actions'], indent=2)}
        Reasoning: {state['reasoning']}
        
        Please provide:
        1. A summary of what was found
        2. What actions were taken and why
        3. The expected impact of these actions
        4. Any follow-up recommendations
        5. Evidence and calculations supporting the decisions
        
        Make this explanation clear and professional for stakeholders.
        """
        
        messages = [
            SystemMessage(content="You are a financial consultant. Provide a clear, professional explanation of the investigation and corrective actions taken."),
            HumanMessage(content=explanation_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["reasoning"] = response.content
        
        return state
    
    def run_investigation(self, mission: str = "Investigate revenue leakage") -> Dict[str, Any]:
        """Run the complete investigation workflow"""
        initial_state = {
            "mission": mission,
            "findings": [],
            "proposals": [],
            "applied_actions": [],
            "reasoning": "",
            "current_step": "",
            "user_input": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log"""
        return self.tools.get_audit_log()
    
    def get_applied_actions(self) -> List[Dict[str, Any]]:
        """Get applied actions"""
        return self.tools.get_applied_actions()
    
    def rollback_action(self, action_id: str) -> Dict[str, Any]:
        """Rollback a specific action"""
        return self.tools.rollback(action_id)
