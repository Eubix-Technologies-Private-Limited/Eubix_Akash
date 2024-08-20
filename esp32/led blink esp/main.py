from machine import Pin
from time import sleep


led = Pin(27,Pin.OUT)

while True:
    led.value(1)
    sleep(0.7)
    led.value(0)
    sleep(0.7)