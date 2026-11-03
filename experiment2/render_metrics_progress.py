"""Overlay verified aggregate results on representative archived throws."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    exp=Path(__file__).resolve().parent
    root=exp/'analysis/final_20261031'
    report=json.loads((root/'heldout/evaluation.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    rows=report['episodes']
    hits=sorted([i for i,r in enumerate(rows) if r['reward_terms'].get('board_impact',0)>0],key=lambda i:rows[i]['impact_error_m'])
    chosen=[hits[len(hits)//2],next(i for i,r in enumerate(rows) if r['outcome']=='environment_hit'),next(i for i,r in enumerate(rows) if r['outcome']=='robot_hit')]
    output=exp.parent/'progress_videos/ppo_metrics_20261031.mp4'
    if output.exists():raise FileExistsError(output)
    fonts=[ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size) for size in (30,24,20)]
    pipe=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    sources=[]
    try:
        for index in chosen:
            row=rows[index];source=root/'heldout/episodes'/f'{index:06d}'/'swing.mp4'
            cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:raise RuntimeError('Empty recorded video')
            duration=len(frames)/fps*4
            for tick in range(round((duration+1.2)*30)):
                frame=frames[min(int(tick/30/4*fps),len(frames)-1)]
                canvas=Image.new('RGB',(1280,720),'#101925');draw=ImageDraw.Draw(canvas)
                draw.text((24,18),'PPO dart throwing | verified progress',font=fonts[0],fill='white')
                canvas.paste(Image.fromarray(frame).resize((960,576)),(0,72))
                lines=[('FINAL TEST: 32 THROWS','#91bdce'),('Board hits: 21/32 (66%)','white'),('Bullseyes: 0/32','#ffbd69'),('Clean flights: 26/32','white'),('Median hit error: 14.6 cm','white'),('(among 21 valid hits)','#9baec2'),('Robot contacts: 4','#ff9292'),('Hand recontacts: 2','#ff9292'),('Other misses: 5','white'),('404,588-step PPO','#91bdce'),('Calibrated motion prior','#91bdce')]
                for j,(line,color) in enumerate(lines):draw.text((974,88+j*42),line,font=fonts[2],fill=color)
                impact=f" | impact error {row['impact_error_m']*100:.1f} cm" if row['impact_error_m'] is not None else ''
                draw.text((24,654),f"Seed {row['seed']} | {row['outcome']}{impact} | 4x slow motion",font=fonts[1],fill='white')
                draw.text((24,688),'Actual saved physics rollout. Bullseye objective remains unmet.',font=fonts[2],fill='#ffbd69')
                pipe.stdin.write(canvas.tobytes())
            sources.append(dict(seed=row['seed'],outcome=row['outcome'],source=str(source)))
        pipe.stdin.close()
        if pipe.wait():raise RuntimeError('Video encoding failed')
    finally:
        if pipe.poll() is None:pipe.terminate();pipe.wait()
    output.with_suffix('.json').write_text(json.dumps(dict(summary=summary,clips=sources,slow_motion_factor=4),indent=2))
    print(output,flush=True)


if __name__=='__main__':main()
