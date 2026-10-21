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
Heartbeat pulse, `print("hello world")`, code fills the screen, the code talks, forms an eye,
the eye looks at you, the camera pulls back to the Mona Lisa on a museum wall.
```
python code_to_monalisa.py --export monalisa.mp4
python code_to_monalisa.py --image full_painting.jpg --eye X Y   # use a bigger image
python code_to_monalisa.py --no-audio
```

## Ideas for next ones
- Logo reveal for clients: code assembles into a brand logo
- Any photo or painting as the final reveal
- Text-to-ASCII intros for reels
