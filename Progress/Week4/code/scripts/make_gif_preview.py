"""Convert a short excerpt of an actual recorded rollout to an inline preview."""
from pathlib import Path
import argparse
import imageio_ffmpeg
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('video',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--start',type=float,default=1.)
p.add_argument('--end',type=float,default=6.)
args=p.parse_args()
if not 0<=args.start<args.end<=60 or args.end-args.start>10:raise ValueError('Preview must be a bounded excerpt of at most10s')
reader=imageio_ffmpeg.read_frames(str(args.video),pix_fmt='rgb24')
metadata=next(reader);fps=metadata['fps'];frames=[]
try:
    for i,raw in enumerate(reader):
        if i/fps>=args.end:break
        if i/fps<args.start or i%2:continue
        frame=Image.frombytes('RGB',metadata['size'],raw).resize((960,540))
        frames.append(frame.quantize(colors=128,dither=Image.Dither.NONE))
finally:reader.close()
if not frames:raise ValueError('Recording has no frames in the preview interval')
args.output.parent.mkdir(parents=True,exist_ok=True)
frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=round(2000/fps),loop=0,optimize=True,disposal=2)
print(f'Saved {len(frames)} actual rollout frames: {args.output}')
