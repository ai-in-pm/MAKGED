import torch
from models import GCNEncoder
from agents import (
    HeadForwardAgent, HeadBackwardAgent,
    TailForwardAgent, TailBackwardAgent,
    DiscussionFacilitator, SummarizerDecisionMaker
)
from config import (
    MAX_DISCUSSION_ROUNDS,
    CONFIDENCE_THRESHOLD,
    VOTING_THRESHOLD
)

class MAKGED:
    def __init__(self):
        # Initialize agents
        self.agents = {
            'hfa': HeadForwardAgent(),
            'hba': HeadBackwardAgent(),
            'tfa': TailForwardAgent(),
            'tba': TailBackwardAgent(),
            'df': DiscussionFacilitator(),
            'sdm': SummarizerDecisionMaker()
        }
        
        # Initialize GCN model
        self.gcn = None  # Will be initialized when data is loaded
        
    def initialize_gcn(self, num_features):
        self.gcn = GCNEncoder(num_features)
        
    def get_graph_embeddings(self, x, edge_index):
        """Get graph structure embeddings using GCN"""
        return self.gcn(x, edge_index)
    
    def get_semantic_embeddings(self, triple):
        """Get semantic embeddings for the triple using OpenAI's embedding model"""
        # Implementation details omitted for brevity
        return torch.randn(768)  # Placeholder
        
    def detect_error(self, triple, graph_context):
        """Main error detection process"""
        print(f"\nAnalyzing triple: {triple}")
        
        # Get embeddings
        graph_emb = self.get_graph_embeddings(graph_context['x'], graph_context['edge_index'])
        semantic_emb = self.get_semantic_embeddings(triple)
        
        # Combined context
        context = {
            'graph_embedding': graph_emb,
            'semantic_embedding': semantic_emb,
            'graph_context': graph_context
        }
        
        # Initial analysis by all agents
        agent_results = {}
        for name, agent in self.agents.items():
            if name not in ['df', 'sdm']:
                confidence, reasoning = agent.analyze_triple(triple, context)
                agent_results[name] = {
                    'confidence': confidence,
                    'reasoning': reasoning
                }
        
        # Check for immediate consensus
        confidences = [result['confidence'] for result in agent_results.values()]
        if all(conf > CONFIDENCE_THRESHOLD for conf in confidences):
            votes = [conf > VOTING_THRESHOLD for conf in confidences]
            is_error = sum(votes) / len(votes) > VOTING_THRESHOLD
            return is_error, "Immediate consensus reached", agent_results
        
        # Initialize discussion
        discussion_history = []
        for round in range(MAX_DISCUSSION_ROUNDS):
            print(f"\nDiscussion Round {round + 1}")
            
            # Facilitate discussion
            discussion_points = self.agents['df'].facilitate_discussion(agent_results)
            discussion_history.append({
                'round': round + 1,
                'points': discussion_points,
                'agent_results': agent_results.copy()
            })
            
            # Agents participate in discussion and potentially update their stance
            for name, agent in self.agents.items():
                if name not in ['df', 'sdm']:
                    confidence, reasoning = agent.participate_in_discussion(discussion_points)
                    agent_results[name] = {
                        'confidence': confidence,
                        'reasoning': reasoning
                    }
            
            # Check for consensus after discussion
            confidences = [result['confidence'] for result in agent_results.values()]
            if all(conf > CONFIDENCE_THRESHOLD for conf in confidences):
                votes = [conf > VOTING_THRESHOLD for conf in confidences]
                is_error = sum(votes) / len(votes) > VOTING_THRESHOLD
                return is_error, f"Consensus reached after {round + 1} rounds", agent_results
        
        # If no consensus reached, let SDM make final decision
        final_decision = self.agents['sdm'].make_decision(discussion_history)
        return None, "No consensus - SDM decision: " + final_decision, agent_results
