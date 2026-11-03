"""Side view of the two IK stance branches, not a walking demonstration."""
import bootstrap
from bootstrap import WEEK
import json,math
import mujoco,numpy as np
from PIL import Image,ImageDraw

m=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'))
m.vis.global_.offwidth,m.vis.global_.offheight=640,720
canvas=Image.new('RGB',(1280,720),'#f7f9fc');draw=ImageDraw.Draw(canvas)
records={}
with mujoco.Renderer(m,height=720,width=640) as renderer:
    for column,(name,file) in enumerate((('Earlier reverse-flexion stance','gait_home.json'),('Corrected flexion stance','natural_home.json'))):
        h=json.loads((WEEK/'model'/file).read_text());d=mujoco.MjData(m);d.qpos[2]=h['root_height']
        for n,v in h['joint_pos'].items():d.qpos[m.joint(n).qposadr[0]]=v
        mujoco.mj_forward(m,d)
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.27];camera.distance=1.0;camera.azimuth=90;camera.elevation=-10
        option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[0]=option.geomgroup[1]=1
        renderer.update_scene(d,camera=camera,scene_option=option)
        canvas.paste(Image.fromarray(renderer.render()),(column*640,0))
        sides={}
        for s in ('l','r'):
            hip=d.xanchor[m.joint(s+'_hip_roll').id];knee=d.xanchor[m.joint(s+'_knee_roll').id];ankle=d.xanchor[m.joint(s+'_ankle_roll').id]
            fraction=(hip[2]-knee[2])/(hip[2]-ankle[2]);line=hip[0]+fraction*(ankle[0]-hip[0])
            sides[s]=dict(knee_forward_of_hip_ankle_line_mm=float((knee[0]-line)*1000),
                          knee_joint_deg=math.degrees(h['joint_pos'][s+'_knee_roll']),ankle_joint_deg=math.degrees(h['joint_pos'][s+'_ankle_roll']))
        records[name]=sides
        draw.rectangle((column*640,0,column*640+640,78),fill='#172032')
        draw.text((column*640+18,16),name,fill='white')
        draw.text((column*640+18,38),'Side view / same geometry, mass, hard joint and torque limits',fill='white')
        draw.text((column*640+18,58),'Static IK pose comparison only; not a trained walking result',fill='white')
        draw.rectangle((column*640,660,column*640+640,720),fill='#172032')
        draw.text((column*640+18,675),f'Left knee angle {sides["l"]["knee_joint_deg"]:.1f} deg / ankle {sides["l"]["ankle_joint_deg"]:.1f} deg',fill='white')
        draw.text((column*640+18,697),f'Knee forward offset: {sides["l"]["knee_forward_of_hip_ankle_line_mm"]:.1f} mm',fill='white')
output=WEEK/'results/quality'
canvas.save(output/'flexion_comparison.png')
(output/'flexion_comparison.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
