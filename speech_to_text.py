import speech_recognition as sr

#README: this file is contains the function in which users can talk to chatbot

def listen_n_transcribe(prompt = "Ask me something!"):
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        # set some of the parameters to make sure pauses 
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2.0
        r.non_speaking_duration = 1.0
        print(prompt)
    
        try:
            audio = r.listen(source, timeout= 5)
            words = r.recognize_google(audio)
        
        #any errors thrown will result in re-prompting
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said. Please try again.")
            listen_n_transcribe()
        
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            listen_n_transcribe()
        
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
            listen_n_transcribe()


    words = r.recognize_google(audio)
    return words #string type

    #problem right now is speech takes really long to transcribe

