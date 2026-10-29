"""Residual PPO environment for contact-based G1 dart throws (SI units)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from inspire_grip import InspireGrip
from inspire_model import NAMES
from experiment1_throw import ARM, make_simple_plan
from trajectory import guidance, release_score


def impact_reward(radius, tip_first, clean):
    if not (tip_first and clean):return -15., False
    bull=radius <= .00635
    return 20.+20.*np.exp(-(radius/.08)**2)+(100. if bull else 0.), bool(bull)


class DartThrowEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':50}
    def __init__(self, randomize=False, record=False, variant='legacy'):
        if variant not in ('legacy','arc_v2'):raise ValueError('Unknown environment variant')
        self.variant=variant
        self.sim=s=InspireGrip();self.m=s.model;self.d=s.data
        self.randomize=randomize;self.record=record;self.frames=[];self.renderer=None
        for n,v in zip(NAMES,s.config['q_grasp']):s.config['arm_pose']['R_'+n[2:]]=v
        self.plan,self.curve=make_simple_plan(s)
        for n,v in zip(ARM,self.plan['arm_keyframes'][1]):s.config['arm_pose'][n]=v
        s.config['arm_pose'].update(waist_yaw_joint=-.025,waist_pitch_joint=-.01)
        self.names=ARM+['waist_yaw_joint','waist_pitch_joint']
        self.jids=np.array([self.m.joint(n).id for n in self.names])
        self.qadr=self.m.jnt_qposadr[self.jids];self.vadr=self.m.jnt_dofadr[self.jids]
        self.scale=np.array([.45,.25,.30,.65,.20,.50,.20,.15,.12])
        # More upward-motion authority, with the same joints and actuator limits.
        # Fixed per run: observation/action layouts remain checkpoint-compatible.
        self.slew=.04
        if variant=='arc_v2':
            self.scale=np.array([.70,.35,.45,1.,.30,.65,.30,.20,.18])
            self.slew=.06
        self.action_space=spaces.Box(-1.,1.,shape=(10,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(73,),dtype=np.float32)
        self.board=self.m.body('dartboard').id;self.board_geom=self.m.geom('board_collision').id
        self.tip_geom=self.m.geom('spare_dart_1_point').id
        self.goal=self.d.site_xpos[self.m.site('bullseye').id].copy();self.goal[0]=3.
        self.robot=set();pelvis=self.m.body('pelvis').id
        for b in range(self.m.nbody):
            parent=b
            while parent:
                if parent==pelvis:self.robot.add(b);break
                parent=int(self.m.body_parentid[parent])
        self.hand_bodies={b for b in self.robot if self.m.body(b).name.startswith('L_')}
        self.jp=np.zeros((3,self.m.nv));self.jr=self.jp.copy()
        self.nominal_mass=self.m.body_mass[s.dart];self.nominal_inertia=self.m.body_inertia[s.dart].copy()
        self.finger_geoms=np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in self.hand_bodies])
        self.nominal_friction=self.m.geom_friction[self.finger_geoms].copy()
        self.reset(seed=0)

    def velocity(self):
        mujoco.mj_jacBodyCom(self.m,self.d,self.jp,self.jr,self.sim.dart)
        return self.jp@self.d.qvel

    def observation(self):
        s,d=self.sim,self.d;r=s.sample()
        return np.r_[d.qpos[self.qadr],d.qvel[self.vadr]/10,
                     d.qpos[s.fq],d.qvel[s.fv],d.xipos[s.dart],self.velocity(),
                     d.xmat[s.dart].reshape(3,3)[:,0],self.goal,
                     np.asarray(r['normal_N'])/5,self.residual,self.previous_action,
                     [self.t,self.commanded,self.released,self.clear_time,self.max_progress,
                      r['relative_drift_m'],d.xipos[s.dart][2],self.mass_scale,self.friction_scale]].astype(np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);s,m,d=self.sim,self.m,self.d
        self.mass_scale=float(self.np_random.uniform(.95,1.05)) if self.randomize else 1.
        self.friction_scale=float(self.np_random.uniform(.85,1.15)) if self.randomize else 1.
        m.body_mass[s.dart]=self.nominal_mass*self.mass_scale
        m.body_inertia[s.dart]=self.nominal_inertia*self.mass_scale
        m.geom_friction[self.finger_geoms]=self.nominal_friction*self.friction_scale
        # Initialize dependent constants after changing physical parameters.
        # mj_setConst overwrites MjData; place the initial grasp afterwards.
        mujoco.mj_setConst(m,d)
        s.reset(perturb=self.np_random.uniform(-.0001,.0001,3) if self.randomize else None)
        s.velocity_target=np.zeros(m.nv);s.acceleration_target=np.zeros(m.nv)
        for _ in range(500):s.step()
        if min(s.sample()['normal_N'])<.02:raise RuntimeError('Initial three-finger grasp failed')
        self.start_time=float(d.time);self.t=0.;self.commanded=False;self.released=False
        self.clear_time=0.;self.release_info=None;self.max_progress=0.;self.residual=np.zeros(9)
        self.previous_action=np.zeros(10);self.outcome=None;self.success=False;self.impact_error=None
        self.start_pos=d.xipos[s.dart].copy();self.frames=[];self.episode_return=0.
        self.terms={};self.clean=True;self.release_bonus_paid=False
        self.max_finger_speed=0.;self.max_limit_violation=0.
        self.command_time=None;self.trajectory=None
        self.arc_potential=guidance(d.xipos[s.dart],self.velocity(),self.goal)['velocity_quality']
        self.peak_height=float(d.xipos[s.dart][2]);self.closest_tip_distance=np.inf
        self.board_crossing=None;self.previous_tip=d.site_xpos[m.site('spare_dart_1_tip').id].copy()
        self.nominal_target=d.qpos.copy()
        if self.record:self.capture()
        return self.observation(),{}

    def add(self,name,value):self.terms[name]=self.terms.get(name,0.)+float(value);return float(value)

    def contact_event(self):
        """Check all robot links, not just the three intended fingertips."""
        s,m,d=self.sim,self.m,self.d;hand_contact=False;hits=[];force=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i];b1,b2=m.geom_bodyid[[c.geom1,c.geom2]]
            if s.dart not in (b1,b2):continue
            mujoco.mj_contactForce(m,d,i,force)
            if force[0]<1e-5:continue
            other=int(b2 if b1==s.dart else b1)
            og=int(c.geom2 if b1==s.dart else c.geom1);dg=int(c.geom1 if b1==s.dart else c.geom2)
            if other in self.hand_bodies:
                hand_contact=True
                if self.released:hits.append(('recontact',c.pos.copy(),False))
                elif other not in s.distal or '_barrel_' not in m.geom(dg).name:
                    hits.append(('bad_grip_contact',c.pos.copy(),False))
            elif other in self.robot:hits.append(('robot_hit',c.pos.copy(),False))
            elif og==self.board_geom:
                tip=d.site_xpos[m.site('spare_dart_1_tip').id]
                hits.append(('board',c.pos.copy(),dg==self.tip_geom and np.linalg.norm(c.pos-tip)<.003))
            else:hits.append(('environment_hit',c.pos.copy(),False))
        # Invalid collisions take precedence over any simultaneous board reward.
        hits.sort(key=lambda h:h[0]=='board')
        return hand_contact,hits

    def step(self,action):
        s,m,d=self.sim,self.m,self.d
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        if action.shape!=(10,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        reward=self.add('smoothness',-.01*np.mean((action[:9]-self.previous_action[:9])**2))
        reward+=self.add('residual_cost',-.002*np.mean(action[:9]**2))
        self.previous_action=action.copy()
        if self.t>=.02 and action[-1]>.6 and not self.commanded:
            self.commanded=True;s.closed=False;self.command_time=self.t
        new=np.clip(action[:9]*self.scale,self.residual-self.slew,self.residual+self.slew)
        old=self.residual.copy();self.residual=new
        for k in range(40):
            phase=min(self.t+2.1,3.8);s.target[:]=self.curve(phase)
            # Freeze the unused right hand at its valid reset pose.
            for n in NAMES:s.target[m.joint('R_'+n[2:]).qposadr[0]]=s.config['arm_pose']['R_'+n[2:]]
            s.target[self.qadr]+=old+(new-old)*(k+1)/40
            s.target[self.qadr]=np.clip(s.target[self.qadr],m.jnt_range[self.jids,0]+.04,m.jnt_range[self.jids,1]-.04)
            s.velocity_target[:]=0;s.acceleration_target[:]=0
            s.velocity_target[self.vadr]=self.curve(phase,1)[self.qadr]+(new-old)/.02
            s.step();self.t=float(d.time-self.start_time)
            mujoco.mj_forward(m,d)
            self.peak_height=max(self.peak_height,float(d.xipos[s.dart][2]))
            tip=d.site_xpos[m.site('spare_dart_1_tip').id].copy()
            if self.released:
                self.closest_tip_distance=min(self.closest_tip_distance,float(np.linalg.norm(tip-self.goal)))
                if self.previous_tip[0]<self.goal[0]<=tip[0] and self.board_crossing is None:
                    fraction=(self.goal[0]-self.previous_tip[0])/(tip[0]-self.previous_tip[0])
                    crossing=self.previous_tip+fraction*(tip-self.previous_tip)
                    self.board_crossing={'position':crossing.tolist(),'error_m':float(np.linalg.norm(crossing[1:]-self.goal[1:]))}
            self.previous_tip=tip
            self.max_finger_speed=max(self.max_finger_speed,float(np.max(np.abs(d.qvel[s.fv]))))
            q=d.qpos[self.qadr];lim=m.jnt_range[self.jids]
            violation=float(np.max(np.maximum(lim[:,0]-q,q-lim[:,1])))
            self.max_limit_violation=max(self.max_limit_violation,violation)
            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning) or violation>.03:
                self.outcome='unstable_or_joint_limit';reward+=self.add('invalid',-25.);break
            hand,hits=self.contact_event()
            if hand:self.clear_time=0.
            else:self.clear_time+=m.opt.timestep
            if self.clear_time>=.01 and not self.released:
                if not self.commanded:
                    self.outcome='premature_slip';reward+=self.add('premature_slip',-15.);break
                self.released=True;v=self.velocity();p=d.xipos[s.dart].copy()
                self.release_info={'time_s':self.t,'position':p.tolist(),'velocity':v.tolist()}
                self.trajectory=guidance(p,v,self.goal)
                self.trajectory['command_to_detach_s']=self.t-self.command_time if self.command_time is not None else None
                if v[0]<.3:
                    self.outcome='backward_or_drop';reward+=self.add('bad_release',-12.);break
                reward+=self.add('clean_release',2.)
                # Bounded ballistic shaping is guidance only; never declares a hit.
                flight=(self.goal[0]-p[0])/v[0]
                predicted=p+v*flight+np.array([0,0,-4.905*flight**2])
                err=np.linalg.norm(predicted[1:]-self.goal[1:])
                if self.variant=='legacy':reward+=self.add('release_prediction',3.*np.exp(-(err/.8)**2))
                else:
                    score,cosine=release_score(self.trajectory,d.xmat[s.dart].reshape(3,3)[:,0],v)
                    reward+=self.add('release_arc',score)
                    self.trajectory['nose_velocity_cosine']=cosine
            if hits:
                kind,pos,tip_first=hits[0]
                self.outcome=kind
                if kind=='board':
                    self.impact_error=float(np.linalg.norm(pos[1:]-self.goal[1:]))
                    clean=self.released and self.commanded and d.xmat[s.dart].reshape(3,3)[0,0]>.5
                    value,self.success=impact_reward(self.impact_error,tip_first,clean)
                    reward+=self.add('board_impact',value)
                else:reward+=self.add(kind,-15. if kind in ('robot_hit','recontact','bad_grip_contact') else -8.)
                break
            if self.released:
                progress=float(np.clip((d.xipos[s.dart][0]-self.start_pos[0])/(self.goal[0]-self.start_pos[0]),0,1))
                if progress>self.max_progress:
                    reward+=self.add('forward_flight',2.*(progress-self.max_progress));self.max_progress=progress
            if self.t>=1. and not self.commanded:
                self.outcome='no_release';reward+=self.add('no_release',-12.);break
        if self.record:self.capture()
        # Only improvement in release readiness is paid; waiting cannot collect
        # repeated positive shaping. This heuristic is not claimed to be PBRS.
        if self.variant=='arc_v2' and not self.released:
            potential=guidance(d.xipos[s.dart],self.velocity(),self.goal)['velocity_quality']
            reward+=self.add('arc_improvement',6.*(potential-self.arc_potential))
            self.arc_potential=potential
        terminated=self.outcome is not None;truncated=self.t>=2.5 and not terminated
        if truncated:reward+=self.add('timeout',-8.);self.outcome='timeout'
        if self.variant=='arc_v2' and (terminated or truncated) and self.released and self.outcome in ('environment_hit','board','timeout'):
            # Measured free-flight approach complements the approximate ballistic
            # teacher; it includes our simulated drag, rotation and contact path.
            reward+=self.add('actual_approach',8./(1.+(self.closest_tip_distance/.5)**2))
        reward+=self.add('finger_speed',-.01*max(0.,self.max_finger_speed-.5)*.02)
        self.episode_return+=reward
        info={'is_success':self.success,'outcome':self.outcome,'impact_error_m':self.impact_error,
              'release':self.release_info,'reward_terms':self.terms.copy(),
              'variant':self.variant,'trajectory':self.trajectory,'peak_height_m':self.peak_height,
              'closest_tip_distance_m':float(self.closest_tip_distance) if np.isfinite(self.closest_tip_distance) else None,
              'board_plane_crossing':self.board_crossing,
              'max_finger_speed_rad_s':self.max_finger_speed,'max_joint_limit_violation_rad':self.max_limit_violation}
        return self.observation(),float(reward),terminated,truncated,info

    def capture(self):
        self.frames.append((self.t,self.d.qpos.copy(),self.d.qvel.copy()))

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.m,480,800)
        camera=mujoco.MjvCamera();camera.lookat[:]=[1.3,0,1.05];camera.distance=4.;camera.azimuth=90;camera.elevation=-10
        option=mujoco.MjvOption();option.geomgroup[3]=0
        self.renderer.update_scene(self.d,camera,scene_option=option)
        return self.renderer.render()

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
