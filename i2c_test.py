# Testing I2C communication between Raspberry Pi and PIC by blinking an LED
# Goal: Pi sends alternating 1 and 0 bits, turning the testing LED connected to the PIC on or off

import smbus2
import time

ADDRESS = 0x08          # PIC I2C address
bus = smbus2.SMBus(1)   # I2C bus 1 on Pi

def send_value(value):
    bus.write_byte(ADDRESS, value)
    print(f"Sent: {value}")

while True:
    send_value(1)  # Send 1
    time.sleep(1)  # Wait 1 second
    send_value(0)  # Send 0
    time.sleep(1)