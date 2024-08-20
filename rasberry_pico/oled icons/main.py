from machine import Pin, I2C
import ssd1306  # Ensure you have this library in Wokwi for OLED
import time

# I2C setup for OLED
i2c = I2C(0, scl=Pin(17), sda=Pin(16))
oled = ssd1306.SSD1306_I2C(128, 64, i2c)  # Adjust if your OLED has a different size

# Setup
button1 = Pin(14, Pin.IN, Pin.PULL_DOWN)
button2 = Pin(0, Pin.IN, Pin.PULL_DOWN)
buzzer = Pin(15, Pin.OUT)
button_increase = Pin(18, Pin.IN, Pin.PULL_DOWN)
button_decrease = Pin(19, Pin.IN, Pin.PULL_DOWN)

# Set initial brightness level
brightness = 128  # Default brightness level (range is typically 0-255)
oled.contrast(brightness)

icon_flik = [
    0b00000000,
    0b00011000,
    0b00011000,
    0b00111100,
    0b00111100,
    0b00011000,
    0b00011000,
    0b00000000
]

# Icons (8x8 for simplicity)
icon_brightness = [
    0b00111100,
    0b01000010,
    0b01000010,
    0b10000001,
    0b10000001,
    0b01000010,
    0b01000010,
    0b00111100
]

icon_power = [
    0b00011000,
    0b00111100,
    0b01111110,
    0b01111110,
    0b01111110,
    0b00111100,
    0b00011000,
    0b00000000
]

icon_buzzer = [
    0b00001000,
    0b00001000,
    0b00111100,
    0b01111110,
    0b01111110,
    0b00111100,
    0b00001000,
    0b00001000
]

# Image data for "A" (replace with your own image data)
bold_letter_a_32x32 = [
    # Top white portion (16 rows)
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************

    # Center portion with "A" (32 rows)
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111100000000011111111111111,  # ****        ****    
    0b11111110000000000000111111111111,  # *****            
    0b11111100001111111100001111111111,  # ****             
    0b11111000011111111110000111111111,  # ****             
    0b11111000111111111111000111111111,  # ***              
    0b11110000111111111111000011111111,  # ****             
    0b11110001111111111111100011111111,  # ***              
    0b11100001111111111111100001111111,  # ****             
    0b11100000000000000000000001111111,  # ********************
    0b11000000000000000000000000111111,  # ********************
    0b11011111111111111111111100011111,  # ***               
    0b10011111111111111111111100001111,  #****               
    0b10011111111111111111111110001111,  #***                
    0b00001111111111111111111110000111,  #****               
    0b00011111111111111111111111000111,  #***                
    0b00011111111111111111111111000111,  #***                
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************
    0b11111111111111111111111111111111,  # ******************

    # Bottom white portion (16 rows)
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111,  # ********************
    #0b11111111111111111111111111111111   # ********************
]
    # ... [Image data unchanged] ...

def show_bold_letter_a_with_brightness():
    
    oled.fill(0)
    x_offset = 48
    y_offset = 0

    pixel_on_density = brightness // 32  # Simulate different brightness levels
    
    for y, row in enumerate(bold_letter_a_32x32):
        for x in range(32):
            if row & (1 << (31 - x)):  # Check if the bit is set
                # Simulate brightness by displaying fewer pixels at lower brightness
                if pixel_on_density >= 8 or (pixel_on_density > 0 and (x + y) % (8 // pixel_on_density) == 0):
                    oled.pixel(x + x_offset, y + y_offset, 1)
    
    oled.show()
def display_changes():
    show_bold_letter_a_with_brightness()
    display_icons()
def display_changes_flik():
    show_bold_letter_a_with_brightness()
    display_icons_flik()

def display_icons():
    # Display brightness icon at position (16, 56)
    display_icon(icon_brightness, 16, 56)
    # Display power icon at position (56, 56)
    display_icon(icon_power, 56, 56)
    # Display buzzer icon at position (96, 56)
    display_icon(icon_buzzer, 96, 56)
    oled.show()

def display_icons_flik():
    # Display brightness icon at position (16, 56)
    display_icon(icon_flik, 16, 56)
    # Display power icon at position (56, 56)
    display_icon(icon_power, 56, 56)
    # Display buzzer icon at position (96, 56)
    display_icon(icon_buzzer, 96, 56)
    oled.show()

def display_icon(icon, x_offset, y_offset):
    for y, row in enumerate(icon):
        for x in range(8):
            if row & (1 << (7 - x)):
                oled.pixel(x + x_offset, y + y_offset, 1)

# Function to display the even bolder letter "A" on OLED
#def show_bold_letter_a_on_oled():
 #   oled.fill(0)
 #   x_offset = 48  # Adjust this to center the letter on the display
  #  y_offset = 16

   # for y, row in enumerate(bold_letter_a_32x32):
    #    for x in range(32):
     #       if row & (1 << (31 - x)):  # Check if the bit is set
      #          oled.pixel(x + x_offset, y + y_offset, 1)
    
    #oled.show()

pb1 = 0
b1 = 0
b2 = 0
oled_on=0
b = 2
buzzer.on()
# Main loop
while True:
    bb2 = button2.value()
    bb1 = button1.value()
    if  bb2:
        b2 = not b2
    
    if pb1 == 0 and bb1 == 1:
        b1 = not b1
        if b1:
            buzzer.on()  # Turn on the buzzer
            oled_on =1
            print("oled power on!!!! :)")
              # Show image on the OLED
        else:
            print("oled power off!!! :(")
            buzzer.off()  # Turn off the buzzer
            oled.fill(0)  # Clear the OLED
            oled.show()  # Update the OLED to clear the display
            oled_on = 0
    if oled_on and  bb1 :
        
        show_bold_letter_a_with_brightness()
        display_icons()
        
    #if pb2 == 0 and bb2 == 1:
     #   b2 = not b2
      #  if b2:
       #     buzzer.on()  # Turn on the buzzer
        #else:
         #   buzzer.off()
    #else:
     #          buzzer.on()  # Turn on the buzzer

    if b2 :
        buzzer.on()

    else:
        buzzer.off()

    if button_increase.value() == 1:  # Button pressed (active-low)
        if brightness < 255:
            brightness += 64
            b+=1
            oled.contrast(brightness)
            #display_icon(icon_brightness, 16, 56)
            #time.sleep(0.1)
            display_changes_flik()
            display_changes()
            print("Brightness level:", b)
            time.sleep(0.1)  # Debounce delay
        #print("Brightness level:", brightness)
    if button_decrease.value() == 1:  # Button pressed (active-low)
        if brightness > 0:
            brightness -= 64  # Decrease brightness
            b-=1
            oled.contrast(brightness)
            #display_changes_flik()
            #display_icon(icon_brightness, 16, 56)
            #time.sleep(0.1)
            display_changes()
            print("Brightness level:", b)
            time.sleep(0.1)  # Debounce delay
        #print("Brightness level:", brightness)#brightness)

    time.sleep(0.1)  # Small delay to debounce
