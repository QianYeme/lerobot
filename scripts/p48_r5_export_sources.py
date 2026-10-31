"""Export r5's four fixed Train reference frames, without choosing target coordinates."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/lerobot-main')
WORK = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT = WORK/'result'
DATA = Path('/root/autodl-tmp/p48_data/lerobot_libero')


def main():
    import av
    import pandas as pd
    from PIL import Image,ImageDraw
    split = json.loads((OUT/'dataset_split_manifest.json').read_text())
    ids = sorted(split['train_episode_ids']);assert len(ids)==35
    chosen = [ids[int(len(ids)*q)] for q in [0,.25,.5,.75]]
    meta = pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    folder = OUT/'prototype_frames';folder.mkdir(exist_ok=True)
    sheet = Image.new('RGB',(512,560),'white');draw = ImageDraw.Draw(sheet);sources=[]
    for n,ep in enumerate(chosen):
        m = meta.loc[meta.episode_index==ep].iloc[0];frame = int(int(m.length)*.25)
        key = 'videos/observation.images.image/'
        path = DATA/'videos/observation.images.image'/f'chunk-{int(m[key+"chunk_index"]):03d}'/f'file-{int(m[key+"file_index"]):03d}.mp4'
        target = round(float(m[key+'from_timestamp'])*10)+frame
        with av.open(str(path)) as container:
            rgb = next(f.to_ndarray(format='rgb24') for i,f in enumerate(container.decode(video=0)) if i==target)
        assert rgb.shape==(256,256,3)
        image = Image.fromarray(rgb);name = f'ep{ep:04d}_frame{frame:04d}_raw.png'
        image.save(folder/name)
        image.resize((768,768),Image.Resampling.NEAREST).save(folder/name.replace('_raw','_display3x'))
        x=n%2*256;y=n//2*280;sheet.paste(image,(x,y+24));draw.text((x+4,y+5),f'ep{ep} frame{frame} RAW RGB',fill='black')
        sources.append(dict(episode=ep,frame=frame,episode_length=int(m.length),
                            camera_key='observation.images.image',width=256,height=256,
                            source_video=str(path),source_video_frame=target,
                            source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            filename='prototype_frames/'+name,flip=False,crop=False,
                            annotation_coordinate_system='native dataset256x256; x right, y down',
                            resize_for_display_only=3,train_membership=True))
    sheet.save(OUT/'prototype_contact_sheet.png')
    (OUT/'prototype_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(sources),flush=True)


if __name__=='__main__':main()
