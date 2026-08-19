# Catboy-vs.-Puppyboy-inator


Onshape Link : 


https://cad.onshape.com/documents/d6c669abeeb3d7413e8d21f8/w/bde522f027b95c0c69872acb/e/c43f71f52c280984713279bd?renderMode=0&uiState=6a39e0cea890183796429e63

____________________ 


<img width="1410" height="2000" alt="catboy vs puppyboy" src="https://github.com/user-attachments/assets/7bf0b90c-8097-426e-bdcd-672405229f23" />
A device that use a camera, machine learning, and hopes and dreams to detect if the registered faces are more likely to be a catboy or a puppyboy :D


# What is the Catboy vs. Puppboy Dectector?

It has:

- camera to capture your face
- buttons to scan your face
- an ML brain (hand-tuned facial-landmark math) ANNDD a trained image classifier voting together
- a screen that displays your verdict as a percentage
- RBG LED that shows you your results as a bonus

_____

#Why It Was Made? 

Well. There is no answer that I could give you that sounds professional or good enough in this readme other than the fact that I really enjoy guessing if people are cat-like or puppy-like. I technically play live furry-guesser with friends in public. 

I learned AI and machine learning in my AP CSA class and decided that yeah, my knowledge can be used to greater purposes such as this. I love making entertaining projects that gets a laugh out of people. 

(lowk im kinda into catboys yo)


# Pictures (☆▽☆)


CAD image:


<img width="474" height="494" alt="image" src="https://github.com/user-attachments/assets/1f4e918d-1549-41fd-859f-7b62b5eda415" />


PCB:


<img width="1195" height="689" alt="image" src="https://github.com/user-attachments/assets/aafb0ef7-5e71-4387-86b1-48d6820deb90" />


<img width="729" height="866" alt="image" src="https://github.com/user-attachments/assets/3c41c299-1bf7-4134-995f-7ed2c42142b7" />




Live Scoring Demo (Application): 



<img width="624" height="640" alt="image" src="https://github.com/user-attachments/assets/62a9901b-c821-4f5b-9b6d-c6925e0debee" />


## BILL OF MATERIALS


| Name | Qty | Unit | Price (USD) | Total (USD) | MOQ | MOQ Price | Source |
| :--- | :---: | :---: | ---: | ---: | :---: | ---: | :--- |
| Seeed XIAO ESP32S3 Sense (Pre-Soldered) | 1 | pcs | $14.90 | $14.90 | 1 | $14.90 | [Order](https://www.seeedstudio.com/Seeed-Studio-XIAO-ESP32S3-Sense-Pre-Soldered-p-6335.html) |
| Grove OLED Display 0.66" (SSD1306) | 1 | pcs | $5.50 | $5.50 | 1 | $5.50 | [Order](https://www.seeedstudio.com/Grove-OLED-Display-0-66-SSD1306-v1-0-p-5096.html) |
| TS-1187A-B-A-B Tactile Switch | 1 | pcs | $0.02 | $0.37 | 20 | $0.37 | [Order](https://www.lcsc.com/product-detail/C318884.html) |
| Adafruit NeoPixel Ring 12x5050 RGB LED | 1 | pcs | $8.95 | $8.95 | 1 | $8.95 | [Order](https://www.adafruit.com/product/1643) |
| Adafruit LiPo Battery 3.7V 500mAh | 1 | pcs | $7.95 | $7.95 | 1 | $7.95 | [Order](https://www.adafruit.com/product/1578) |
| JST S2B-PH-SM4-TB Battery Connector (SMD) | 1 | pcs | $0.12 | $0.12 | 1 | $0.12 | [Order](https://www.lcsc.com/product-detail/C295747.html) |
| PCB | 1 | pcs | $0.40 | $2.00 | 5 | $2.00 | [Order](https://jlcpcb.com) |
| PLA Housing | 1 | pcs | - | - | - | - | Self-Print / Local |
| M3 x 10mm Socket Cap Head Screws | 4 | pcs | $0.47 | $1.88 | 4 | $1.88 | [Order](https://accu-components.com/us/metric-cap-head-screws/16004-SSCF-M3-10-12-9) |
| M3 x 5mm Socket Cap Head Screws | 5 | pcs | $0.31 | $1.86 | 6 | $1.86 | [Order](https://accu-components.com/us/metric-cap-head-screws/16001-SSCF-M3-5-12-9) |

____

# BUILD & ASSEMBLY
___
# Required Tools
____
1) Soldering iron + solder
2) Tweezers
3) 3D Printer
4) Screwdriver for M3 screws
5) Computer with Arduino IDE + Python installed

# Assembly Directions:
______
1) Solder the ESP32C3. camera, buttons, LEDs, and battery onto the PCB following the schematic and layout
2) Check for incomplete connections
3) Mount the camera so it faces outward
4) Route the battery leads and place the LiPo into battery compartment. Check for polarity!
5) Align PCB into mounting posts then press it into place
6) Position top case onto the bottom case
7) Tighten screws into place
8) Install all the software libraries in Arduino
9) Set WiFi credentials and companion server IP at the top of the .ino sketch
10) Compile firmware and check for errors (shouldn't be any, if there is, check syntax for your edits)
11) On your laptop, start the companion Python server(python server.py). This runs the ML
12) Power on the device and press scan

After that confirm:
the display, results/score, battery life, wifi connection, wifi behavior, button, and LED

and then.. YOU'RE SET!

lmk if you're a catboy

# CREDITS:
- Onshape (for CAD)
- KiCad (schematic + PCB)
- MediaPipe (facial landmark extraction)
- Google Teachable Machine (classifier training)
Every friend who let us take a photo of their face for "science"

# DEPENDENCIES

Dependencies

You need these libraries (add via Arduino Library Manager):

- ArduinoJson (by Benoit Blanchon)
- WiFi (Built-in, ESP32 board package)
- HTTPClient (Built-in, ESP32 board package)
- (Built-in, ESP32 board package)
 
Python side (companion server):

- mediapipe
- opencv-python
- fastapi
- uvicorn
- numpy

# references
my big, beautiful brain for thinking of this idea.

and the catboys on pinterest
