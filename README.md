# ascii-animations

Animations made only with code: every frame is drawn with text characters.

## Setup
```
pip install -r requirements.txt
```

## 1. Eye to Heart (16:9)
An ASCII eye opens, looks around, blinks, then the camera dives into the pupil and finds a beating heart.
```
python eye_to_heart.py                       # live window
python eye_to_heart.py --export eye.mp4      # render video
python eye_to_heart.py --export eye_v.mp4 --size 1080x1920
```

## 2. Code to Mona Lisa (9:16, with sound)
`print("hello world")`, code fills the screen, the code talks and forms an eye. The iris
forms out of code, the eye searches around, blinks and stares at you, then dissolves into the
Mona Lisa's eye and the camera pulls back to the painting on a museum wall.
```
python code_to_monalisa.py --export monalisa.mp4
python code_to_monalisa.py --image full_painting.jpg --eye X Y   # use a bigger image
python code_to_monalisa.py --no-audio
```

## 3. 2ND SUN stage visual concept (9:16, cut to music)
One continuous code transformation, no cuts: code fills the page and forms the logo, the
star inside it starts to shine, two hands drawn in code reach in and a light blooms where they
meet on the star, on the drop it explodes into a sun, the sun becomes the iris of one big eye
that floats and looks around on the beat. The camera dives into the pupil, the two dots in it
grow into two moons, a rocket comes in from out of frame and burns A SPACE ODYSSEY in as it
passes, then comes back close to the camera and its giant trail paints the date: FRI OCT 30.
Cut to the CHIHIRO guitar cover (110 BPM): its bar-5 strum lands on the explosion at 9.52s
and every pulse follows its beat. `--start` picks where to cut the track (negative starts it late).
```
python second_sun.py --music chihiro.mp3 --export second_sun.mp4
```

## 4. The Curse of Being Human (9:16, red and black, cut to a spoken-word track)
One continuous camera, no cuts, every scene turns into the next: red code types across the screen
and becomes Ultron, eyes glowing, his head turning slowly; his eyes flare and the fire spreads from them
over him and burns him away in sparks to an x-ray skull with a cassette for a brain;
the tape unspools down to money on a string; the string snaps, the money falls and the camera
falls after it into a cell with hands gripping the bars; the prison is inside an eye, the eye is one of a wall of eyes that dart about and blink,
turn to look at you, look away, then close; the eye with the prison closes last and its lid line becomes the
seam of a closed ring box held in two hands; the fingers lift the lid on the hit, an Esc key inside, never
pressed: everything goes dark but the key's light, then that too. Ultron, the fire, x-ray, money, the eye with bars and the ring box
come from reference images in `assets/`, drawn in code. The lyrics
type out as subtitles.
```
python curse_of_being_human.py --music voice.mp4 --export curse.mp4
python curse_of_being_human.py --music voice.mp4 --no-lyrics      # visuals only
```

## Ideas for next ones
- Logo reveal for clients: code assembles into a brand logo
- Any photo or painting as the final reveal
- Text-to-ASCII intros for reels
