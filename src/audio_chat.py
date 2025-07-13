import speech_recognition as sr

def transcribe_audio(listen=False):
    while listen:
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)

            if text:
                return text
            else:
                print("No speech detected, please try again.")
                return None
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None