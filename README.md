# Multi-Agent Knowledge Graph Error Detection (MAKGED)

A real-time simulation framework that employs six AI agents leveraging Large Language Models (LLMs) for enhanced error detection in knowledge graphs. The system integrates graph structure embeddings (via GCN) and semantic embeddings (via LLMs) while demonstrating multi-agent discussion and voting mechanisms.

## Features

- **Multi-Agent Architecture**: Six specialized AI agents working collaboratively:
  - Head Forward Agent (HFA)
  - Head Backward Agent (HBA)
  - Tail Forward Agent (TFA)
  - Tail Backward Agent (TBA)
  - Discussion Facilitator (DF)
  - Summarizer & Decision Maker (SDM)

- **Hybrid Embeddings**: Combines both structural and semantic information:
  - Graph Convolutional Networks (GCNs) for structural embeddings
  - Large Language Models for semantic embeddings

- **Structured Discussion Process**: Multi-round discussion mechanism for consensus building
- **Fallback Mechanisms**: Graceful error handling for API issues
- **Real-time Analysis**: Dynamic error detection in knowledge graph triples

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- API keys for:
  - OpenAI
  - Anthropic
  - Cohere
  - Google Cloud AI Platform
  - EmergenceAI

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Multi-Agent-Knowledge-Graph-Error-Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
GROQ_API_KEY="your-groq-key"
GOOGLE_API_KEY="your-google-key"
COHERE_API_KEY="your-cohere-key"
EMERGENCEAI_API_KEY="your-emergenceai-key"
```

## Project Structure

- `config.py`: Configuration settings and API key management
- `models.py`: Graph neural network models for structural embeddings
- `agents.py`: Implementation of the six AI agents
- `makged.py`: Core framework implementation
- `main.py`: Example usage and demonstration

## Usage

Run the example script to see the framework in action:
```bash
python main.py
```

This will:
1. Create a sample knowledge graph
2. Initialize the MAKGED framework
3. Analyze a test triple using all agents
4. Demonstrate the multi-agent discussion process
5. Output the final decision with reasoning

## Agent Roles

### Head Forward Agent (HFA)
Evaluates subgraphs where the entity serves as the head, focusing on forward relationships.

### Head Backward Agent (HBA)
Evaluates subgraphs where the entity serves as the tail, analyzing backward relationships.

### Tail Forward Agent (TFA)
Analyzes subgraphs where the tail entity is in a forward position.

### Tail Backward Agent (TBA)
Analyzes subgraphs where the tail entity is in a backward position.

### Discussion Facilitator (DF)
Oversees structured discussion rounds and maintains fairness in agent debates.

### Summarizer & Decision Maker (SDM)
Synthesizes findings and provides final decisions with comprehensive reasoning.

## Discussion Process

1. Initial Analysis:
   - Each agent independently analyzes the triple
   - If all agents agree, decision is finalized

2. Discussion Rounds:
   - Up to three rounds of structured debate
   - Agents refine positions using supporting evidence
   - Discussion Facilitator guides the process

3. Decision Making:
   - Majority vote after discussion
   - SDM resolves ties with reasoning

## Error Detection Example

```python
from makged import MAKGED

# Initialize framework
makged = MAKGED()

# Test triple
triple = ("Huawei Honor 10", "network support", "5G")

# Detect errors
is_error, decision_msg, agent_results = makged.detect_error(triple, graph_context)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Cohere for language models
- Google Cloud AI Platform
- PyTorch Geometric team for GNN implementations
