"""Orchestrator agent (Week 12).

Top-level LangGraph node that dispatches to specialists (Investigator,
PolicyInterpreter, VisionAnalyst, FraudAuditor) in parallel, then merges
state and routes to ContextSynthesizer → Adjudicator → Critic → Communicator.
"""
