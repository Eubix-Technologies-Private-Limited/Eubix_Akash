from pygame import mixer
mixer.init() 
sound=mixer.Sound("beep.mp3")
sound.play()