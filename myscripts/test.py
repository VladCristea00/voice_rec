import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 135)
engine.say('Vă rog selectați ce doriți să modificați: 1. Notificări, 2. Limbă, 3.Locație, 4.Politică de confidențialitate, 5.Deconectare')
engine.runAndWait()

