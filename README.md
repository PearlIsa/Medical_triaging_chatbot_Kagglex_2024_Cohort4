# Medical_triaging_chatbot_Kagglex_2024_Cohort4
## Project Overview

This project implements a Medical Triage Chatbot designed to assist users in assessing the severity of their symptoms and providing appropriate medical advice. The chatbot uses advanced natural language processing and speech recognition technologies to interact with users through both text and voice inputs.

## Key Features

1. **Symptom Assessment**: Analyzes user-reported symptoms and asks relevant follow-up questions.
2. **Severity Classification**: Categorizes symptoms based on severity and urgency.
3. **Personalized Recommendations**: Provides tailored advice ranging from self-care tips to urgent medical attention.
4. **Multi-modal Input**: Accepts both text and speech inputs for user convenience.
5. **Empathetic Responses**: Generates context-aware, empathetic responses to enhance user experience.

## Technologies Used

- **Python**: Primary programming language
- **TensorFlow & PyTorch**: For deep learning model implementation
- **Hugging Face Transformers**: Utilizing pre-trained models for NLP tasks
- **Wav2Vec2**: For speech-to-text conversion
- **Gemma (2B version)**: For natural language understanding and generation
- **Gradio**: For creating an interactive web interface

## Dataset Integration

The project incorporates five key datasets:

1. **Persona-Chat**: Enhances conversational fluency
2. **Medalpaca/Medical_Meadow_MEDQA**: Provides clinical decision-making capabilities
3. **Medical-Diagnosis-Synthetic**: Maps symptoms to potential diagnoses
4. **LibriSpeech ASR**: Improves speech-to-text processing
5. **Common Voice**: Enhances speech recognition capabilities

## Model Architecture

### Speech-to-Text Model
- **Wav2Vec2**: Converts audio inputs to text
- Fine-tuned on medical conversations for improved accuracy in medical terminology

### Natural Language Understanding Model
- **Gemma (2B version)**: Processes text inputs and generates responses
- Fine-tuned using LoRA (Low-Rank Adaptation) for efficient adaptation to medical dialogue

## Key Components

1. **Symptom Dictionary**: Contains detailed information about various symptoms, including clarifying questions and self-care advice.
2. **Conversation Flow Generator**: Creates dynamic, context-aware conversations between the user and the chatbot.
3. **Severity Classifier**: Assesses the urgency of reported symptoms based on user responses.
4. **Recommendation Engine**: Provides appropriate advice based on symptom severity and user context.

## Demo

The project includes a Gradio-based demo interface for easy interaction with the chatbot. To launch the demo:

```python
python gradio_demo.py

## Current Limitations and Future Work

The model's performance is currently limited by the small dataset used for initial testing.
Future improvements include:
1. Expanding the training dataset for better generalization
2. Fine-tuning the Wav2Vec2 model on domain-specific medical conversations
3. Enhancing the conversation flow with more complex branching logic



## Disclaimer
This chatbot is designed for informational purposes only and does not replace professional medical advice. Users should consult with healthcare professionals for proper medical diagnosis and treatment.
## Contributors
Pear Isa

## Acknowledgments

1. Hugging Face for providing pre-trained models and datasets
3. The open-source community for various libraries and tools used in this project

