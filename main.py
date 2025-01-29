import torch
import networkx as nx
from makged import MAKGED

def create_sample_knowledge_graph():
    # Create a sample knowledge graph
    G = nx.DiGraph()
    
    # Add some nodes (entities)
    entities = [
        "Huawei Honor 10",
        "5G",
        "4G",
        "Kirin 970",
        "Android 8.1",
        "6GB RAM",
        "128GB Storage"
    ]
    
    for entity in entities:
        G.add_node(entity)
    
    # Add edges (relationships)
    edges = [
        ("Huawei Honor 10", "network support", "4G"),
        ("Huawei Honor 10", "processor", "Kirin 970"),
        ("Huawei Honor 10", "operating system", "Android 8.1"),
        ("Huawei Honor 10", "RAM", "6GB RAM"),
        ("Huawei Honor 10", "storage", "128GB Storage")
    ]
    
    for head, relation, tail in edges:
        G.add_edge(head, tail, relation=relation)
    
    return G

def convert_graph_to_pytorch_geometric(G):
    # Convert networkx graph to PyTorch Geometric format
    num_nodes = len(G.nodes())
    
    # Create node features (simple one-hot encoding for demonstration)
    x = torch.eye(num_nodes)
    
    # Create edge index
    edge_index = []
    for edge in G.edges():
        src = list(G.nodes()).index(edge[0])
        dst = list(G.nodes()).index(edge[1])
        edge_index.append([src, dst])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    return {'x': x, 'edge_index': edge_index}

def main():
    # Create sample knowledge graph
    print("Creating sample knowledge graph...")
    kg = create_sample_knowledge_graph()
    
    # Convert to PyTorch Geometric format
    graph_data = convert_graph_to_pytorch_geometric(kg)
    
    # Initialize MAKGED
    print("\nInitializing MAKGED framework...")
    makged = MAKGED()
    makged.initialize_gcn(num_features=len(kg.nodes()))
    
    # Test triple to analyze (intentionally incorrect to demonstrate error detection)
    test_triple = ("Huawei Honor 10", "network support", "5G")
    
    print("\nStarting error detection process...")
    is_error, decision_msg, agent_results = makged.detect_error(test_triple, graph_data)
    
    print("\nResults:")
    print(f"Is Error: {is_error}")
    print(f"Decision Message: {decision_msg}")
    print("\nAgent Reasonings:")
    for agent, result in agent_results.items():
        print(f"\n{agent.upper()}:")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    main()
