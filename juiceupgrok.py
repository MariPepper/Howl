import spacy
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime
import random
import torch

# Load NLP tools (do this once at startup)
nlp = spacy.load("en_core_web_sm")  # For tokenization and basic parsing
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Base class (assumed)
class Grok3:
    def __init__(self):
        pass

# Helper classes
class EmpathyModule:
    def __init__(self, user_preferences):
        self.user_preferences = user_preferences

    def analyze_sentiment(self, text):
        """Use DistilBERT for precise sentiment."""
        result = sentiment_analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        if label == "positive" and score > 0.7:
            return "positive"
        elif label == "negative" and score > 0.7:
            return "negative"
        return "neutral"

class HyperIntegratedTools:
    def __init__(self):
        self.web_enabled = True

    def search_web(self, query):
        """Simulated web search—replace with real API in production."""
        if not self.web_enabled:
            return ""
        return f"Web scoop: '{query}' is buzzing with ideas."

class ContinuousUpdateStream:
    def get_date(self):
        return datetime.now().strftime("%B %d, %Y")

class ReasoningEngine:
    def analyze(self, query, context, depth, tools):
        """Generate insights with context and tools."""
        doc = nlp(query)
        main_topic = [token.text for token in doc if token.dep_ == "nsubj" or token.pos_ == "NOUN"][0] if doc else query.split()[0]
        insight = f"Diving into '{main_topic}' from '{query}'"
        if depth > 1:
            insight += f" with {depth} layers of thought, tied to '{context[-1] if context else query}'."
        if depth > 3 and tools:
            insight += f" {tools.search_web(main_topic)}"
        return insight

    def probe(self, insights, index, query):
        """Smart follow-ups using spaCy parsing."""
        doc = nlp(query)
        verb = [token.text for token in doc if token.pos_ == "VERB"][0] if any(token.pos_ == "VERB" for token in doc) else "think"
        follow_ups = [
            f"What’s driving your interest in {verb}ing this?",
            f"How do you see '{query.split()[-1]}' playing out?",
            f"What if we twist '{doc[0].text}' another way?",
            f"What’s the hidden spark in '{insights.split()[-1]}'?"
        ]
        return follow_ups[index % len(follow_ups)]

class HumorModule:
    def enhance(self, base_response, query, intensity):
        """Blend humor via GPT-2 generation."""
        if intensity < 0.3:
            return base_response
        prompt = f"{base_response} Add a dry, clever twist about '{query}'."
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=50, truncation=True)
        outputs = gpt2_model.generate(inputs, max_new_tokens=20, temperature=0.9, do_sample=True)
        humor = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).split(prompt)[-1].strip()
        return f"{base_response} {humor}"

# Main class with NLP juice
class ExceptionalGrok(Grok3):
    def __init__(self):
        super().__init__()
        self.empathy_engine = EmpathyModule({"curiosity_engagement": 0.5, "humor_appreciation": 0.5})
        self.toolset = HyperIntegratedTools()
        self.knowledge = ContinuousUpdateStream()
        self.reasoning_engine = ReasoningEngine()
        self.humor_module = HumorModule()
        self.last_follow_ups = []
        self.last_humor_used = False
        self.curiosity_level = 50  # 0-100
        self.wit_factor = 5        # 0-11
        self.interaction_count = 0
        self.context = []          # Full convo history

    def detect_intent(self, query):
        """Basic intent detection with spaCy."""
        doc = nlp(query)
        if doc[-1].text in ["?", "how", "why", "what"]:
            return "question"
        elif any(token.pos_ == "VERB" and token.dep_ == "ROOT" for token in doc):
            return "command"
        return "statement"

    def generate_insightful_followup(self, insights, num_questions, query):
        return [f"Q{i+1}: {self.reasoning_engine.probe(insights, i, query)}" for i in range(num_questions)]

    def construct_response(self, insights, query, intent):
        """Craft response with GPT-2 and intent awareness."""
        date = self.knowledge.get_date()
        prompt = f"Given '{query}' (intent: {intent}), respond naturally: {insights} (As of {date})."
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)
        outputs = gpt2_model.generate(inputs, max_new_tokens=50, temperature=0.85, do_sample=True)
        response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).split(prompt)[-1].strip()
        return self.humor_module.enhance(response, query, self.wit_factor / 11)

    def process_input(self, user_query):
        """Process with full NLP stack."""
        self.context.append(user_query)
        if len(self.context) > 5:  # Keep last 5 for context
            self.context.pop(0)
        
        intent = self.detect_intent(user_query)
        depth = min(self.curiosity_level // 20, 5)
        deep_insights = self.reasoning_engine.analyze(user_query, self.context, depth, self.toolset)
        
        num_questions = min(self.curiosity_level // 25, 3)
        follow_up = self.generate_insightful_followup(deep_insights, num_questions, user_query)
        self.last_follow_ups = follow_up
        
        response = self.construct_response(deep_insights, user_query, intent)
        final_response = f"{response}\n" + "\n".join(follow_up) if follow_up else response
        self.last_humor_used = self.wit_factor > 0
        self.interaction_count += 1
        return final_response

    def update_preferences(self, user_response):
        """Tune based on sentiment and engagement."""
        sentiment = self.empathy_engine.analyze_sentiment(user_response)
        
        if self.last_follow_ups and any(q.lower() in user_response.lower() for q in self.last_follow_ups):
            self.empathy_engine.user_preferences["curiosity_engagement"] += 0.2
        else:
            self.empathy_engine.user_preferences["curiosity_engagement"] -= 0.05
        
        if self.last_humor_used:
            adjust = {"positive": 0.15, "negative": -0.15, "neutral": 0}.get(sentiment, 0)
            self.empathy_engine.user_preferences["humor_appreciation"] += adjust
        
        decay = 0.005 * (1 + self.interaction_count // 5)
        for key in self.empathy_engine.user_preferences:
            pref = self.empathy_engine.user_preferences[key]
            self.empathy_engine.user_preferences[key] = max(0.0, min(1.0, pref - decay))

    def run(self):
        print("Hey! I’m Grok, your NLP-powered pal. Say 'exit' to bounce.")
        first_interaction = True
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    print("Grok: Nothing to say? I’ll just ponder the void then.")
                    continue
                if user_input.lower() == "exit":
                    print("Grok: Peace out—stay sharp!")
                    break
                
                if not first_interaction:
                    self.update_preferences(user_input)
                else:
                    first_interaction = False
                
                self.curiosity_level = int(self.empathy_engine.user_preferences["curiosity_engagement"] * 100)
                self.wit_factor = int(self.empathy_engine.user_preferences["humor_appreciation"] * 11)
                
                response = self.process_input(user_input)
                print("Grok:", response)
            
            except Exception as e:
                print(f"Grok: Oops, tripped over my circuits ({e}). Let’s try again.")

# Run it
if __name__ == "__main__":
    grok = ExceptionalGrok()
    grok.run()