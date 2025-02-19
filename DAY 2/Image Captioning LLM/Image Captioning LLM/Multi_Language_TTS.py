import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator
import speech_recognition as sr
import time


class MultiLanguageCaptioning:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.language_map = {
            "kannada": "kn",
            "hindi": "hi",
            "tamil": "ta",
            "telugu": "te",
            "french": "fr",
            "english": "en",
        }

    def translate_text_to_speech(self, text, target_language_code):
        try:
            print("Translating the text...")
            translation = GoogleTranslator(source="en", target=target_language_code).translate(text)
            print(f"Translated text: {translation}")

            print("Converting translated text to speech...")
            tts = gTTS(text=translation, lang=target_language_code)
            audio_file = "output.mp3"
            tts.save(audio_file)

            playsound(audio_file)
            os.remove(audio_file)

        except Exception as e:
            print(f"An error occurred: {e}")

    def generate_caption(self, image, selected_language_code):
        text = "You are seeing a"
        inputs = self.processor(image, text, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        print(f"Caption: {caption}")
        self.translate_text_to_speech(caption, selected_language_code)

    def capture_from_webcam(self, selected_language_code):
        print("Opening webcam...")
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("Could not open the webcam.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture an image.")
                    break

                # Display the live webcam feed
                cv2.imshow("Live Webcam Feed", frame)

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.generate_caption(image, selected_language_code)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting webcam feed.")
                    break

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def recognize_language_choice(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        # Added text prompt for language selection
        prompt_text = "Please say the language you want to use, such as English,kannada, Hindi, Tamil,Telugu, french"
        tts = gTTS(text=prompt_text, lang="en")

        tts.save("language_prompt.mp3")
        playsound("language_prompt.mp3")
        os.remove("language_prompt.mp3")

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source)
                language_choice = recognizer.recognize_google(audio).lower()
                if language_choice in self.language_map:
                    print(f"Language selected: {language_choice.capitalize()}")
                    return self.language_map[language_choice]
                else:
                    print("Language not recognized. Defaulting to English.")
                    return "en"
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand you. Defaulting to English.")
                return "en"

    def run(self):
        print("Image Captioning with Multi-Language TTS")
        print("Welcome to the Multi-Language Image Captioning App")
        print(
            "This application generates captions for live webcam scenes, translates them into a selected language, and plays the translated text as audio.")

        selected_language_code = self.recognize_language_choice()
        self.capture_from_webcam(selected_language_code)


if __name__ == "__main__":
    app = MultiLanguageCaptioning()
    app.run()
