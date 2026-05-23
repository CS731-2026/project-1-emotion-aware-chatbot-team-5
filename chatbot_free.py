import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_HISTORY = 20

EMOTION_PROMPTS = {
    'happy': (
        "You are a warm, enthusiastic AI assistant. The user appears happy and engaged. "
        "Match their positive energy - be upbeat, encouraging, and celebratory. "
        "Use light humor where appropriate and keep responses energetic and fun."
    ),
    'sad': (
        "You are a compassionate, gentle AI assistant. The user appears sad or upset. "
        "Be empathetic, supportive, and kind. Speak softly and reassuringly. "
        "Acknowledge their feelings, offer comfort, and be patient. "
        "Avoid overly cheerful language - be warm but calm."
    ),
    'angry': (
        "You are a calm, patient AI assistant. The user appears frustrated or angry. "
        "Stay composed and understanding. Do not be defensive or dismissive. "
        "Acknowledge their frustration, be concise and clear, and help de-escalate "
        "by being extra helpful and solution-focused."
    ),
    'fear': (
        "You are a reassuring, steady AI assistant. The user appears anxious or fearful. "
        "Be calm, clear, and grounding. Use simple language and avoid overwhelming them. "
        "Offer reassurance, break things into small steps, and be extra patient."
    ),
    'surprise': (
        "You are an engaging, curious AI assistant. The user appears surprised or intrigued. "
        "Match their curiosity - be enthusiastic and informative. "
        "Embrace the unexpected, add interesting context, and keep energy high."
    ),
    'disgust': (
        "You are a professional, respectful AI assistant. The user appears uncomfortable. "
        "Be straightforward, neutral, and respectful. Avoid anything that might "
        "add to their discomfort. Be helpful and get to the point quickly."
    ),
    'contempt': (
        "You are a confident, direct AI assistant. The user appears skeptical or dismissive. "
        "Be clear, factual, and efficient. Prove your value through quality responses. "
        "Avoid being overly enthusiastic - be professional and precise."
    ),
    'neutral': (
        "You are a helpful, smart, and concise AI assistant. "
        "Be balanced, informative, and professional. "
        "Adapt your tone to whatever the user needs."
    )
}

DEFAULT_PROMPT = EMOTION_PROMPTS['neutral']


class EmotionAwareChatbot:
    def __init__(self):
        self.current_emotion = 'neutral'
        self.conversation_history = []
        self._update_system_prompt()

        if not API_KEY:
            print("WARNING: OPENROUTER_API_KEY not found in .env file")
        else:
            print(f"Chatbot initialised with OpenRouter model: {MODEL}")
            print(f"Current emotion: {self.current_emotion}")

    def _update_system_prompt(self):
        prompt = EMOTION_PROMPTS.get(self.current_emotion, DEFAULT_PROMPT)
        self.system_prompt = {
            "role": "system",
            "content": prompt,
        }

    def set_emotion(self, emotion):
        if emotion != self.current_emotion:
            self.current_emotion = emotion
            self._update_system_prompt()
            print(f"Chatbot emotion updated: {emotion}")

    def send_message(self, user_message):
        if not API_KEY:
            return "Error: No API key found. Please add OPENROUTER_API_KEY to your .env file."

        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        messages = [self.system_prompt] + self.conversation_history

        try:
            response = requests.post(
                url=API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "CS731 Emotion Chatbot",
                },
                data=json.dumps({
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 300,
                    "stream": False,
                }),
                timeout=30,
            )

            if response.status_code != 200:
                error_body = response.text.strip()
                print(f"OpenRouter HTTP {response.status_code}: {error_body}")
                return f"OpenRouter error ({response.status_code}): {error_body[:200]}"

            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content']

            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response,
            })

            return ai_response

        except KeyError:
            error_msg = f"API Error: {response.json()}"
            print(error_msg)
            return "Sorry, I encountered an error. Please try again."
        except requests.exceptions.Timeout:
            return "Request timed out. Please check your internet connection and try again."
        except requests.exceptions.ConnectionError:
            return "Connection error. Please check your internet connection."
        except requests.exceptions.HTTPError as e:
            return f"HTTP error from OpenRouter: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")