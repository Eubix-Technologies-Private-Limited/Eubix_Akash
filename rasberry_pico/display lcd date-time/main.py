from machine import Pin
import utime
import time

# Initialize the LCD using the gpio_lcd library
from gpio_lcd import GpioLcd

# Create the LCD object using the gpio_lcd library
lcd = GpioLcd(rs_pin=Pin(15),
              enable_pin=Pin(14),
              d4_pin=Pin(9),
              d5_pin=Pin(8),
              d6_pin=Pin(7),
              d7_pin=Pin(6),
              num_lines=2, num_columns=16)

# Create a custom character (happy face)
happy_face = bytearray([0x1F,
  0x1F,
  0x1B,
  0x1B,
  0x1F,
  0x1F,
  0x1B,
  0x1B])
lcd.custom_char(0, happy_face)

# Initialize the LCD display
lcd.clear()

# Get user input (name and current date/time)
name = input("Enter your name: ")
year = int(input("Enter the current year (e.g., 2024): "))
month = int(input("Enter the current month (1-12): "))
day = int(input("Enter the current day (1-31): "))
hour = int(input("Enter the current hour (0-23): "))
minute = int(input("Enter the current minute (0-59): "))
second = int(input("Enter the current second (0-59): "))
date = "{:02}/{:02}/{:04}".format(day, month, year)

# Display the clock, name, and happy face
while True:
    # Format the time string
    time_str = "{:02}:{:02}:{:02}".format(hour, minute, second)
    
    # Clear the display
    lcd.clear()
    
    # Display the current time and name on the first line
    lcd.move_to(0, 0)
    lcd.putstr(time_str + " " + name)
    
    # Display the date and happy face on the second line
    lcd.move_to(0, 1)
    lcd.putstr(date + " ")
    lcd.putchar(chr(0))  # Display the happy face
    
    # Update the time
    utime.sleep(1)
    second += 1
    
    if second >= 60:
        second = 0
        minute += 1
        if minute >= 60:
            minute = 0
            hour += 1
            if hour >= 24:
                hour = 0
