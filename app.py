# ✅ Optimized Triage Chatbot Code for Hugging Face Space (NVIDIA T4 GPU)
# Covers: Memory optimizations, 4-bit quantization, lazy loading, FAISS caching, faster inference, safe Gradio UI
# Includes: Proper Gradio history handling, response cleaning, safety checks

import os
import time
import torch
import logging
import gradio as gr
import psutil
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
import aiohttp
import asyncio
from googlesearch import search
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PearlyBot")

# ===========================
# 🧠 SECRETS MANAGER
# ===========================
class SecretsManager:
    @staticmethod
    def setup():
        load_dotenv()
        creds = {
            'HF_TOKEN': os.getenv('HF_TOKEN'),
        }
        if creds['HF_TOKEN']:
            login(token=creds['HF_TOKEN'])
            logger.info("🔐 Logged in to Hugging Face")
        return creds

# ===========================
# 🧠 MEDICAL CHATBOT CORE
# ===========================
class MedicalTriageBot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.vector_store = None
        self.chunk_size = 512
        self.chunk_overlap = 128
        self.num_relevant_chunks = 5
        self.last_interaction_time = time.time()
        self.interaction_cooldown = 1.0
        self.nhs_api_url = "https://api.nhs.uk/conditions/"
        self.safety_phrases = [
            "999", "111", "emergency", "GP", "NHS",
            "consult a doctor", "seek medical attention"
        ]
        self.triage_levels = {
            'Emergency': [
                'pediatric', 'stroke', 'cardiac', 'unconscious', 'suicidal',
                'psychotic', 'seizure', 'overdose', 'cyanosis'
            ],
            'Urgent': [
                'pregnancy', 'fracture', 'asthma', 'withdrawal',
                'postpartum', 'harm thoughts', 'panic attack'
            ],
            'GP Care': [
                'eczema', 'ocd', 'depression', 'anxiety', 'PTSD',
                'bipolar', 'migraine', 'chronic'
            ],
            'Self-Care': [
                'cold', 'rash', 'stress', 'mild', 'situational',
                'acne', 'insomnia'
            ]
        }
        self.current_case = None
        self.location_services = {
            "London": {"A&E": ["St Thomas' Hospital", "Royal London Hospital"],
                      "Urgent Care": ["UCLH Urgent Care Centre"]},
            "Manchester": {"A&E": ["Manchester Royal Infirmary"]}
        }
        self.base_questions = [
            ("duration", "How long have you experienced this?"),
            ("severity", "On a scale of 1-10, how severe is it?"),
            ("emergency_signs", "Any difficulty breathing, chest pain, or confusion?")
        ]

    async def query_nhs_api(self, symptom: str):
        """Dynamic API query for latest NHS guidelines"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.nhs_api_url}{symptom}") as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"NHS API Error: {e}")
            return None

    def web_fallback(self, query: str):
        """Google search fallback for NHS resources"""
        try:
            NHS_SITES = ["site:nhs.uk", "site:gov.uk"]
            return [j for j in search(f"{query} {' '.join(NHS_SITES)}", num=3, stop=3)]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def get_medical_context(self, query):
        """Hybrid context retrieval system"""
        try:
            # 1. Try local knowledge base
            local_context = self.vector_store.similarity_search(query, k=2)
            if len(local_context) < 1:
                raise ValueError("No local results")
                
            # 2. Fallback to NHS API
            api_data = asyncio.run(self.query_nhs_api(query))
            if api_data:
                return self._parse_api_response(api_data)
                
            # 3. Final fallback to web search
            web_results = self.web_fallback(query)
            return "\n".join(web_results[:2])
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""

    def _parse_api_response(self, api_data):
        """Structure NHS API response for LLM consumption"""
        return f"""
        [NHS Direct Guidelines]
        Condition: {api_data.get('name', '')}
        Symptoms: {', '.join(api_data.get('symptoms', []))}
        Treatment: {api_data.get('treatment', '')}
        Last Updated: {api_data.get('dateModified', '')}
        """

    def schedule_knowledge_updates(self):
        """Weekly index rebuild with fresh data"""
        scheduler = BackgroundScheduler()
        scheduler.add_job(self.build_medical_index, 'interval', weeks=1)
        scheduler.start()

    @torch.inference_mode()
    def generate_safe_response(self, message, history):
        """Enhanced with dynamic crisis handling"""
        # Emergency detection
        if any(kw in message.lower() for kw in ['suicid', 'end it all', 'kill myself']):
            return self._mental_health_crisis_response()
            
        try:
            # Dynamic context handling
            context = self.get_medical_context(message)

    def _determine_triage_level(self, context):
        context_lower = context.lower()
        for level, keywords in triage_levels.items():
            if any(kw in context_lower for kw in keywords):
                return level
        return 'GP Care'

    def _parse_medical_context(self, context):
        components = {
            'advice': [], 'red_flags': [], 
            'timeframe': '24 hours', 'special_considerations': []
        }
        
        current_section = None
        for line in context.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                if key == 'immediate actions':
                    current_section = 'advice'
                    components['advice'].append(value.strip())
                elif key == 'red flags':
                    current_section = 'red_flags'
                    components['red_flags'].append(value.strip())
                elif key == 'action timeline':
                    components['timeframe'] = value.strip()
                elif key == 'cultural considerations':
                    current_section = 'special_considerations'
                    components['special_considerations'].append(value.strip())
            elif current_section:
                components[current_section].append(line)
        
        # Convert lists to strings
        for key in components:
            if isinstance(components[key], list):
                components[key] = '\n- '.join(components[key])
        return components

    def setup_model(self, model_name="google/gemma-7b-it"):
        """Initialize the medical AI model with 4-bit quantization"""
        if self.model is not None:
            return
        
        logger.info("🚀 Initializing medical AI model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(base_model, peft_config)
        logger.info("✅ Medical AI model ready")

    def setup_rag_system(self):
        logger.info("📚 Initializing medical knowledge base")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        if not os.path.exists("medical_index/index.faiss"):
            self.build_medical_index()
            
        self.vector_store = FAISS.load_local(
            "medical_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def build_medical_index(self):
        medical_knowledge = {
            # ===== EMERGENCY CONDITIONS (Call 999) =====
            "emergency_pediatric.txt": """
            Triage Level: Emergency
            Conditions: Pediatric Respiratory Distress, Head Injury
            Key Symptoms:
            - Blue lips/tongue (cyanosis)
            - Stridor or wheezing at rest
            - Repeated vomiting post-head injury
            - Unconsciousness >1 minute
            Immediate Actions:
            1. Call 999 immediately
            2. Maintain airway if trained
            3. Monitor breathing rate
            Cultural Considerations:
            - Provide gender-matched responders if requested
            - Accommodate religious head coverings during assessment""",
    
            "emergency_mental_health.txt": """
            Triage Level: Emergency
            Conditions: Suicidal Crisis, Psychotic Episode
            Key Symptoms:
            - Expressing concrete suicide plan
            - Hallucinations commanding harm
            - Severe self-injury with blood loss
            - Catatonic state
            Immediate Actions:
            1. Call 999 if immediate danger
            2. Remove potential weapons
            3. Stay with patient until help arrives
            Special Considerations:
            - Avoid physical restraint unless absolutely necessary
            - Use non-judgmental language
            Red Flags:
            - Talking about being a burden
            - Sudden calm after depression
            - Giving away possessions""",
    
            # ===== URGENT CARE CONDITIONS (A&E/UTC) =====
            "urgent_mental_health.txt": """
            Triage Level: Urgent
            Conditions: Panic Attacks, Self-Harm
            Key Symptoms:
            - Hyperventilation >30 minutes
            - Superficial self-harm injuries
            - Acute anxiety with derealization
            Action Timeline: Within 4 hours
            While Waiting:
            - Practice grounding techniques
            - Use ice cubes for sensory focus
            - Track panic duration/frequency
            Escalate If:
            - Chest pain develops
            - Dissociation persists >1 hour
            - Urges escalate""",
    
            "urgent_pregnancy.txt": """
            Triage Level: Urgent
            Conditions: Pregnancy Complications
            Key Symptoms:
            - Vaginal bleeding + abdominal pain
            - Reduced fetal movement
            - Sudden swelling + headache
            Action Timeline: Within 2 hours
            Cultural Protocols:
            - Offer female clinician if preferred
            - Respect modesty requests
            Red Flags:
            - Fluid leakage + fever
            - Visual disturbances
            - Contractions <37 weeks""",
    
            # ===== GP CARE CONDITIONS =====
            "gp_mental_health.txt": """
            Triage Level: GP Care
            Conditions: Anxiety, Depression, PTSD
            Key Symptoms:
            - Persistent low mood >2 weeks
            - Panic attacks 2+/week
            - Sleep disturbances + fatigue
            - Avoidance behaviors
            Action Timeline: 72 hours
            Management Strategies:
            - Keep mood/symptom diary
            - Practice 4-7-8 breathing
            - Maintain routine activities
            Red Flags:
            - Social withdrawal >1 week
            - Weight loss >5% in month
            - Suicidal ideation""",
    
            # ===== SELF-CARE CONDITIONS =====
            "selfcare_mental_health.txt": """
            Triage Level: Self-Care
            Conditions: Mild Anxiety, Stress
            Key Symptoms:
            - Situational anxiety
            - Work-related stress
            - Mild sleep difficulties
            Management:
            - Practice box breathing
            - Limit caffeine/alcohol
            - Use worry journal
            - Progressive muscle relaxation
            Escalate If:
            - Symptoms persist >2 weeks
            - Panic attacks develop
            - Daily functioning impaired""",
    
            # ===== PEDIATRIC MENTAL HEALTH =====
            "pediatric_mental_health.txt": """
            Triage Level: GP Care
            Conditions: Childhood Anxiety, School Refusal
            Key Symptoms:
            - School avoidance >3 days
            - Somatic complaints (stomachaches)
            - Nightmares/bedwetting regression
            Action Timeline: 1 week
            Parent Guidance:
            - Maintain consistent routine
            - Validate feelings without reinforcement
            - Use gradual exposure techniques
            Red Flags:
            - Food restriction
            - Self-harm marks
            - Social isolation >1 week""",
    
            # ===== CULTURAL MENTAL HEALTH =====
            "cultural_mental_health.txt": """
            Triage Level: GP Care
            Conditions: Culturally-Specific Presentations
            Key Symptoms:
            - Somatic complaints without medical cause
            - Religious preoccupation/guilt
            - Migration-related stress
            Cultural Considerations:
            - Use trained interpreters
            - Consider spiritual assessments
            - Respect family hierarchy
            Management:
            - Community support referrals
            - Culturally-adapted CBT
            - Family involvement""",
    
            # ===== CHRONIC CONDITIONS =====
            "chronic_mental_health.txt": """
            Triage Level: GP Care
            Conditions: Bipolar, OCD, Eating Disorders
            Key Symptoms:
            - Manic episodes >4 days
            - Compulsions >1hr/day
            - BMI <18.5 with body dysmorphia
            Monitoring:
            - Mood tracking charts
            - Compulsion frequency log
            - Weekly weight checks
            Crisis Signs:
            - Rapid speech + sleeplessness
            - Food restriction >24hrs
            - Contamination fears preventing eating""",
    
            # ===== PERINATAL MENTAL HEALTH =====
            "perinatal_mental_health.txt": """
            Triage Level: Urgent
            Conditions: Postpartum Depression, Psychosis
            Key Symptoms:
            - Intrusive harm thoughts
            - Detachment from baby
            - Visual/auditory hallucinations
            Action Timeline: 24 hours
            Safety Measures:
            - Partner supervision
            - Remove harmful objects
            - Breastfeeding support
            Red Flags:
            - Planning infant harm
            - Extreme paranoia
            - Refusing sleep""",
    
            # ===== ADDICTION & SUBSTANCE ===== 
            "addiction_care.txt": """
            Triage Level: Urgent
            Conditions: Withdrawal, Overdose Risk
            Key Symptoms:
            - Seizure history + alcohol use
            - IV drug use + fever
            - Opioid constricted pupils
            Immediate Actions:
            1. Call 111 for withdrawal management
            2. Monitor breathing rate
            3. Provide naloxone if available
            Danger Signs:
            - Jaundice + abdominal pain
            - Chest pain + stimulant use
            - Hallucinations + tremor"""
        }
    
        # Keep existing index building logic
        os.makedirs("medical_knowledge", exist_ok=True)
        for filename, content in medical_knowledge.items():
            with open(f"medical_knowledge/{filename}", "w") as f:
                f.write(content)
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128
        )
        
        documents = []
        for text in medical_knowledge.values():
            documents.extend(text_splitter.split_text(text))
        
        vector_store = FAISS.from_texts(
            documents,
            self.embeddings,
            metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
        )
        vector_store.save_local("medical_index")

    def _mental_health_crisis_response(self):
        """Immediate crisis intervention"""
        return """🚨 **Emergency Mental Health Support**
1. Call 999 immediately
2. Text SHOUT to 85258 (24/7 crisis text line)
3. Stay on chat - I'll help you connect to services

You're not alone. Let's get through this together."""

    def get_medical_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=2)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"Context error: {e}")
            return ""

    @torch.inference_mode()
    def generate_safe_response(self, message, history):
        try:
            # Add case initialization check
            if self.current_case is None:
                context = self.get_medical_context(message)
                self._initialize_case(context)
                return self._next_question()
                
            context = self.get_medical_context(message)
            triage_level = self._determine_triage_level(context)
            components = self._parse_medical_context(context)
            
            response_templates = {
                'Emergency': (
                    "🚨 **Emergency Alert**\n"
                    "{advice}\n\n"
                    "⚠️ **Immediate Action Required:**\n"
                    "- Call 999 NOW if:\n{red_flags}\n"
                    "🌍 **Special Considerations:**\n{special_considerations}"
                ),
                'Urgent': (
                    "⚠️ **Urgent Care Needed**\n"
                    "Visit A&E within {timeframe} if:\n{red_flags}\n\n"
                    "🩺 **While Waiting:**\n{advice}\n\n"
                    "🌍 **Cultural Notes:**\n{special_considerations}"
                ),
                'GP Care': (
                    "📅 **GP Consultation**\n"
                    "Book appointment within {timeframe}\n\n"
                    "💡 **Self-Care Advice:**\n{advice}\n\n"
                    "⚠️ **Red Flags:**\n{red_flags}"
                ),
                'Self-Care': (
                    "🏡 **Self-Care Management**\n"
                    "{advice}\n\n"
                    "⚠️ **Seek Help If:**\n{red_flags}"
                )
            }
            
            response = response_templates[triage_level].format(**components)
            
            # Add safety netting
            if not any(phrase in response.lower() for phrase in self.safety_phrases):
                response += "\n\nIf symptoms persist, please contact NHS 111."
            
            return response[:500]  # Maintain length limit
    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Please contact NHS 111 directly for urgent medical advice."

    def _initialize_case(self, message):
        context = self.get_medical_context(message)
        self.current_case = {
            "symptoms": self._detect_symptoms(context),
            "responses": {},
            "stage": "investigation",
            "step": 0,
            "location": None,
            "guidelines": context
        }

    def _detect_symptoms(self, context):
        return list(set(
            line.split(":")[0].strip() 
            for line in context.split("\n") 
            if ":" in line
        ))

    def _next_question(self):
        questions = self.base_questions + self._generate_custom_questions()
        if self.current_case["step"] < len(questions):
            return questions[self.current_case["step"]][1]
        self.current_case["stage"] = "location"
        return "Could you share your UK postcode for local service recommendations?"

    def _generate_custom_questions(self):
        return [(f"custom_{i}", q.split(":")[1].strip()) 
                for q in self.current_case["guidelines"].split("\n")
                if "?" in q]

    def _handle_investigation(self, message, history):
        self.current_case["responses"][self.base_questions[self.current_case["step"]][0]] = message
        self.current_case["step"] += 1
        return self._next_question()

    def _handle_location(self, message):
        self.current_case["location"] = self._get_location(message)
        self.current_case["stage"] = "recommendation"
        return self._final_recommendation()

    def _final_recommendation(self):
        action = self._determine_action()
        location_info = self._get_location_services()
        self.current_case = None
        return f"{action}\n\n{location_info}"

    def _determine_action(self):
        if self._is_emergency():
            return "🆘 Call 999 immediately. I can stay on the line with you."
        if self._needs_gp():
            return "📅 Please book a GP appointment. Would you like me to help with that?"
        return "🏥 Visit your nearest urgent care centre:"

    def _get_location_services(self):
        if not self.current_case["location"]:
            return "Find local services: https://www.nhs.uk/service-search"
        return "\n".join([
            f"{service_type}: {', '.join(services)}" 
            for service_type, services in 
            self.location_services.get(self.current_case["location"], {}).items()
        ])

    def _is_emergency(self):
        return any(keyword in self.current_case["guidelines"] 
                for keyword in ["999", "emergency", "stroke"])
    
    def _needs_gp(self):
        return any(keyword in self.current_case["guidelines"]
                for keyword in ["GP", "appointment", "persistent"])
    
    def _get_location(self, postcode):
        return "London" if postcode.startswith("L") else "Manchester" 
        
    

    def _build_prompt(self, message, history):
        conversation = "\n".join([f"User: {user}\nAssistant: {bot}" for user, bot in history[-3:]])
        context = self.get_medical_context(message)
        triage_level = self._determine_triage_level(context)
        
        return f"""<start_of_turn>system
Triage Level: {triage_level}
Context:
{context}
Conversation History:
{conversation}
Response Guidelines:
1. Use {triage_level} response template
2. Include safety netting
3. Consider cultural factors
4. Maintain NHS protocols
<end_of_turn>
<start_of_turn>user
{message}
<end_of_turn>
<start_of_turn>assistant"""
       

# ===========================
# 💬 SAFE GRADIO INTERFACE
# ===========================
def create_medical_interface():
    bot = MedicalTriageBot()
    bot.setup_model()
    bot.setup_rag_system()


    def handle_conversation(message, history):
        try:
            # Handle GP booking requests
            if "book gp" in message.lower():
                return history + [(message, "Redirecting to GP booking system...")]
            
            # Handle location input
            if any(word in message.lower() for word in ["postcode", "zip code", "location"]):
                return history + [(message, "Please enter your UK postcode:")]
            
            # Normal symptom processing
            response = bot.generate_safe_response(message, history)
            return history + [(message, response)]
        
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            return history + [(message, "System error - please refresh the page")]

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# NHS Triage Assistant")
        gr.HTML("""<div class="emergency-banner">🚨 In emergencies, always call 999 immediately</div>""")
        
        with gr.Row():
            chatbot = gr.Chatbot(
                value=[("", "Hello! I'm Pearly, your digital assistant. How can I help you today?")],
                height=500,
                label="Medical Triage Chat"
            )
            
        with gr.Row():
            message_input = gr.Textbox(
                placeholder="Describe your symptoms...",
                label="Your Message",
                max_lines=3
            )
            submit_btn = gr.Button("Send", variant="primary")
            
        clear_btn = gr.Button("Clear History")
        
        # Event handlers
        message_input.submit(
            handle_conversation,
            [message_input, chatbot],
            [chatbot]
        ).then(lambda: "", None, [message_input])
        
        submit_btn.click(
            handle_conversation,
            [message_input, chatbot],
            [chatbot]
        ).then(lambda: "", None, [message_input])
        
        clear_btn.click(
            lambda: [("", "Hello! I'm Pearly, your digital assistant. How can I help you today?")],
            None,
            [chatbot]
        )

    return interface

# ===========================
# 🚀 LAUNCH APPLICATION
# ===========================
if __name__ == "__main__":
    SecretsManager.setup()
    medical_app = create_medical_interface()
    medical_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )