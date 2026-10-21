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
star inside it starts to shine, two hands drawn in code reach in and meet on the star, on
the drop it explodes into a sun, the sun becomes the iris of one big eye that looks around on
the beat, the camera dives into the pupil, an old brick phone drops in spinning (ray-traced into
characters) and its screen lights up, then it breaks into code and the code forms the logo again.
Timed to 126 BPM with the drop 9.52s in; `--start` picks where to cut the track.
```
python second_sun.py --music track.mp3 --start 236.32 --handle @you --export second_sun.mp4
```

## Ideas for next ones
- Logo reveal for clients: code assembles into a brand logo
- Any photo or painting as the final reveal
- Text-to-ASCII intros for reels
