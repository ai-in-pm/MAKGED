import logging
from abc import ABC, abstractmethod
import openai
import anthropic
import google.cloud.aiplatform as aiplatform
import cohere
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name
        self.confidence = 0.0
        self.reasoning = ""
        
    @abstractmethod
    def analyze_triple(self, triple, context):
        pass
    
    @abstractmethod
    def participate_in_discussion(self, other_agents_reasoning):
        pass

class HeadForwardAgent(BaseAgent):
    def __init__(self):
        super().__init__("Head Forward Agent")
        try:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
        
    def analyze_triple(self, triple, context):
        if not self.client:
            logger.warning("OpenAI client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Analyze the following knowledge graph triple from a head-forward perspective:
            Triple: {triple}
            Context: {context}
            Focus on how the head entity relates to other entities in forward direction.
            Is this triple likely to be correct? Provide reasoning."""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.reasoning = response.choices[0].message.content
            self.confidence = 0.9
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

    def participate_in_discussion(self, other_agents_reasoning):
        if not self.client:
            logger.warning("OpenAI client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Based on the discussion points raised by other agents:
            {other_agents_reasoning}
            
            Revise your analysis if necessary. Consider the points raised by other agents and provide updated reasoning."""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.reasoning = response.choices[0].message.content
            self.confidence = 0.9
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

class HeadBackwardAgent(BaseAgent):
    def __init__(self):
        super().__init__("Head Backward Agent")
        try:
            self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None
        
    def analyze_triple(self, triple, context):
        if not self.client:
            logger.warning("Anthropic client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Analyze the following knowledge graph triple from a head-backward perspective:
            Triple: {triple}
            Context: {context}
            Focus on how other entities relate to the head entity."""
            
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.reasoning = response.content[0].text
            self.confidence = 0.85
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

    def participate_in_discussion(self, other_agents_reasoning):
        if not self.client:
            logger.warning("Anthropic client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Based on the discussion points raised by other agents:
            {other_agents_reasoning}
            
            Revise your analysis if necessary. Consider the points raised by other agents and provide updated reasoning."""
            
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.reasoning = response.content[0].text
            self.confidence = 0.85
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

class TailForwardAgent(BaseAgent):
    def __init__(self):
        super().__init__("Tail Forward Agent")
        try:
            self.client = cohere.Client(COHERE_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            self.client = None
        
    def analyze_triple(self, triple, context):
        if not self.client:
            logger.warning("Cohere client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Analyze the following knowledge graph triple from a tail-forward perspective:
            Triple: {triple}
            Context: {context}
            Focus on how the tail entity relates to other entities in forward direction."""
            
            response = self.client.chat(
                message=prompt,
                model='command'
            )
            
            self.reasoning = response.text
            self.confidence = 0.88
        except Exception as e:
            logger.error(f"Error in Cohere API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

    def participate_in_discussion(self, other_agents_reasoning):
        if not self.client:
            logger.warning("Cohere client not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Based on the discussion points raised by other agents:
            {other_agents_reasoning}
            
            Revise your analysis if necessary. Consider the points raised by other agents and provide updated reasoning."""
            
            response = self.client.chat(
                message=prompt,
                model='command'
            )
            
            self.reasoning = response.text
            self.confidence = 0.88
        except Exception as e:
            logger.error(f"Error in Cohere API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

class TailBackwardAgent(BaseAgent):
    def __init__(self):
        super().__init__("Tail Backward Agent")
        try:
            aiplatform.init(project="your-project-id")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Google AI Platform: {e}")
            self.initialized = False
        
    def analyze_triple(self, triple, context):
        if not self.initialized:
            logger.warning("Google AI Platform not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Analyze the following knowledge graph triple from a tail-backward perspective:
            Triple: {triple}
            Context: {context}
            Focus on how other entities relate to the tail entity."""
            
            model = aiplatform.TextGenerationModel.from_pretrained("text-bison@002")
            response = model.predict(prompt)
            
            self.reasoning = response.text
            self.confidence = 0.87
        except Exception as e:
            logger.error(f"Error in Google AI Platform API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

    def participate_in_discussion(self, other_agents_reasoning):
        if not self.initialized:
            logger.warning("Google AI Platform not initialized. Using fallback analysis.")
            return 0.5, "API unavailable - using fallback analysis"
            
        try:
            prompt = f"""Based on the discussion points raised by other agents:
            {other_agents_reasoning}
            
            Revise your analysis if necessary. Consider the points raised by other agents and provide updated reasoning."""
            
            model = aiplatform.TextGenerationModel.from_pretrained("text-bison@002")
            response = model.predict(prompt)
            
            self.reasoning = response.text
            self.confidence = 0.87
        except Exception as e:
            logger.error(f"Error in Google AI Platform API call: {e}")
            self.reasoning = "API error occurred"
            self.confidence = 0.5
            
        return self.confidence, self.reasoning

class DiscussionFacilitator(BaseAgent):
    def __init__(self):
        super().__init__("Discussion Facilitator")
        try:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
        
    def analyze_triple(self, triple, context):
        return 0.0, ""
        
    def facilitate_discussion(self, agent_reasonings):
        if not self.client:
            logger.warning("OpenAI client not initialized. Using simplified facilitation.")
            return "Discussion facilitation unavailable due to API issues."
            
        try:
            prompt = f"""As a Discussion Facilitator, analyze the following agent reasonings and guide the discussion:
            Agent Reasonings: {agent_reasonings}
            Identify key points of agreement and disagreement, and suggest focus areas for the next round."""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return "Discussion facilitation encountered an error."

    def participate_in_discussion(self, other_agents_reasoning):
        pass  # DiscussionFacilitator does not participate in discussion

class SummarizerDecisionMaker(BaseAgent):
    def __init__(self):
        super().__init__("Summarizer & Decision Maker")
        try:
            self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None
        
    def analyze_triple(self, triple, context):
        return 0.0, ""
        
    def make_decision(self, discussion_history):
        if not self.client:
            logger.warning("Anthropic client not initialized. Using simplified decision making.")
            return "Decision making unavailable due to API issues."
            
        try:
            prompt = f"""As the Summarizer & Decision Maker, analyze the following discussion history and make a final decision:
            Discussion History: {discussion_history}
            Provide a clear decision with comprehensive reasoning."""
            
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            return "Decision making encountered an error."

    def participate_in_discussion(self, other_agents_reasoning):
        # SummarizerDecisionMaker does not participate in discussion
        return 0.0, ""
