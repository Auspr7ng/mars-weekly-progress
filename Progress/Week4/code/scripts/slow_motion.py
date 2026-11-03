"""Half-speed playback export, visibly labelled; original rollout is unchanged."""
import argparse
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
reader=imageio_ffmpeg.read_frames(str(args.source),pix_fmt='rgb24');info=next(reader)
writer=imageio_ffmpeg.write_frames(str(args.output),size=info['size'],fps=info['fps']/2,codec='libx264',macro_block_size=1);writer.send(None)
try:
    for raw in reader:
        frame=Image.frombytes('RGB',info['size'],raw);draw=ImageDraw.Draw(frame)
        draw.rectangle((8,52,365,78),fill='#172032');draw.text((16,58),'0.5x PLAYBACK / original physics time shown above',fill='white')
        writer.send(np.asarray(frame))
finally:reader.close();writer.close()
print(args.output)
