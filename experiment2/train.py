"""Train/resume SB3 PPO and preserve every attempted throw as states + video."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import time
from collections import deque,Counter
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback,CallbackList
from env import DartThrowEnv


def normalizer_for(model):
    model=Path(model)
    if model.name.startswith('rl_model_'):
        return model.with_name('rl_model_vecnormalize_'+model.stem.removeprefix('rl_model_')+'.pkl')
    return model.parent/'normalize.pkl'


class RunStatus(BaseCallback):
    def __init__(self,root):super().__init__();self.root=root;self.recent=deque(maxlen=100);self.completed=0
    def write_status(self):
        policy_device=str(next(self.model.policy.parameters()).device)
        optimizer_devices=sorted({str(v.device) for state in self.model.policy.optimizer.state.values() for v in state.values() if torch.is_tensor(v) and v.ndim>0})
        status={'timesteps':self.num_timesteps,'policy_device':policy_device,'optimizer_tensor_devices':optimizer_devices,
                'workers':self.training_env.num_envs,'physics_device':'cpu','updated_utc':datetime.now(timezone.utc).isoformat()}
        temp=self.root/'runtime.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(self.root/'runtime.json')
        if self.recent:
            rows=list(self.recent)
            def med(key):
                values=[r.get('trajectory',{}).get(key) for r in rows if r.get('trajectory') and r['trajectory'].get(key) is not None]
                return float(np.median(values)) if values else None
            metrics={'timesteps':self.num_timesteps,'completed_episodes':self.completed,'window':len(rows),
                     'outcomes':dict(Counter(r.get('outcome') for r in rows)),
                     'bullseyes':sum(bool(r.get('is_success')) for r in rows),
                     'median_velocity_gap_m_s':med('velocity_gap_m_s'),
                     'median_vertical_deficit_m_s':med('vertical_velocity_deficit_m_s'),
                     'median_predicted_board_error_m':med('predicted_board_error_m')}
            temp=self.root/'progress_metrics.tmp';temp.write_text(json.dumps(metrics,indent=2));temp.replace(self.root/'progress_metrics.json')
            for key in ('median_velocity_gap_m_s','median_vertical_deficit_m_s','median_predicted_board_error_m'):
                if metrics[key] is not None:self.logger.record('throw/'+key,metrics[key])
    def _on_training_start(self):self.write_status()
    def _on_rollout_start(self):self.write_status()
    def _on_step(self):
        for done,info in zip(self.locals.get('dones',()),self.locals.get('infos',())):
            if done:self.recent.append(info);self.completed+=1
        return not (self.root/'STOP').exists()


class Archive(gym.Wrapper):
    def __init__(self,env,folder):
        super().__init__(env);self.folder=Path(folder);self.number=0
        self.folder.mkdir(parents=True,exist_ok=True)
    def step(self,action):
        result=self.env.step(action)
        if result[2] or result[3]:self.save(result[4])
        return result
    def save(self,info):
        s=self.env;folder=self.folder/f'{self.number:06d}';self.number+=1
        folder.mkdir(exist_ok=False)
        (folder/'result.json').write_text(json.dumps(info,indent=2))
        np.savez_compressed(folder/'states.npz',time=[v[0] for v in s.frames],qpos=[v[1] for v in s.frames],qvel=[v[2] for v in s.frames])
        # Render archived states, restoring the terminal state for Gym/Monitor.
        final_q=s.d.qpos.copy();final_v=s.d.qvel.copy()
        pipe=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','800x480','-r','50','-i','-',
                               '-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'swing.mp4')],stdin=subprocess.PIPE)
        try:
            for _,q,v in s.frames:
                s.d.qpos[:]=q;s.d.qvel[:]=v;mujoco.mj_forward(s.m,s.d)
                pipe.stdin.write(s.render().tobytes())
            pipe.stdin.close()
            if pipe.wait():raise RuntimeError('Video encoding failed')
        finally:
            if pipe.poll() is None:pipe.terminate();pipe.wait()
            s.d.qpos[:]=final_q;s.d.qvel[:]=final_v;mujoco.mj_forward(s.m,s.d)


def factory(folder,rank,randomize,variant='legacy'):
    def create():
        return Monitor(Archive(DartThrowEnv(randomize=randomize,record=True,variant=variant),Path(folder)/'episodes'/f'worker_{rank}'),str(Path(folder)/f'monitor_{rank}.csv'))
    return create


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1_000_000)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--rollout',type=int,default=512);p.add_argument('--randomize',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--evaluate',type=Path);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    p.add_argument('--checkpoint-every',type=int,default=2048)
    p.add_argument('--variant',choices=('legacy','arc_v2'),default=None)
    p.add_argument('--reset-critic',action='store_true',help='Reinitialize only the value estimator after changing rewards')
    p.add_argument('--learning-rate',type=float,default=None)
    p.add_argument('--entropy',type=float,default=None)
    p.add_argument('--exploration-std',type=float,default=None,help='Explicit new action std when adapting to changed rewards')
    p.add_argument('--desaturate-actor',action='store_true',help='Calibrate saturated mean-action output rows to <=0.8 at reset; retain actor features')
    args=p.parse_args();torch.set_num_threads(1)
    source=args.evaluate or args.resume
    if args.variant is None:
        config_dir=source.parent.parent if source and source.parent.name=='checkpoints' else (source.parent if source else None)
        config=json.loads((config_dir/'config.json').read_text()) if config_dir and (config_dir/'config.json').exists() else {}
        args.variant=config.get('variant') or 'legacy'
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    if min(args.steps,args.workers,args.rollout,args.episodes)<1:raise ValueError('Counts must be positive')
    if args.rollout*args.workers<2:raise ValueError('PPO needs at least two rollout samples')
    root=Path(__file__).parent/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    root.mkdir(parents=True);(root/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    source_dir=Path(__file__).parent;repo=source_dir.parent;manifest={}
    for name in ('experiment2/env.py','experiment2/train.py','experiment2/trajectory.py','experiment1_throw.py','inspire_grip.py','inspire_model.py','inspire_grip/policy.json','dart_model.py','dart_profile.json'):
        path=repo/name;manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    workers=1 if args.evaluate else args.workers
    vector=(DummyVecEnv if workers==1 else SubprocVecEnv)([factory(root,i,args.randomize,args.variant) for i in range(workers)])
    if source:
        vector=VecNormalize.load(str(normalizer_for(source)),vector)
    else:vector=VecNormalize(vector,norm_obs=True,norm_reward=False,clip_obs=10.)
    vector.seed(args.seed)
    if args.evaluate:
        vector.training=False;policy=PPO.load(str(source),env=vector,device=args.device,seed=args.seed)
        obs=vector.reset();reports=[]
        while len(reports)<args.episodes:
            action,_=policy.predict(obs,deterministic=True);obs,rewards,dones,info=vector.step(action)
            if dones[0]:reports.append(info[0])
        # Terminal observations are numpy arrays; keep outcome metrics only.
        reports=[{k:v for k,v in r.items() if k not in ('terminal_observation','episode')} for r in reports]
        (root/'evaluation.json').write_text(json.dumps({'seed':args.seed,'episodes':reports,'bullseye_rate':sum(r['is_success'] for r in reports)/len(reports)},indent=2))
    else:
        policy=PPO.load(str(source),env=vector,device=args.device,seed=args.seed,n_steps=args.rollout,batch_size=min(256,args.rollout*workers)) if source else PPO('MlpPolicy',vector,device=args.device,seed=args.seed,
            n_steps=args.rollout,batch_size=min(64,args.rollout*workers),learning_rate=3e-4,n_epochs=10,
            gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.005,target_kl=.03,
            policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128]),'log_std_init':-.7},verbose=1)
        def actor_digest():
            digest=hashlib.sha256()
            for module in (policy.policy.mlp_extractor.policy_net,policy.policy.action_net):
                for parameter in module.parameters():digest.update(parameter.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        actor_before=actor_digest()
        features_before={k:v.detach().clone() for k,v in policy.policy.mlp_extractor.policy_net.state_dict().items()}
        calibration=None
        if args.desaturate_actor:
            obs=vector.reset()
            with torch.no_grad():
                tensor,_=policy.policy.obs_to_tensor(obs)
                before=policy.policy.get_distribution(tensor).distribution.mean
                gain=torch.clamp(.8/before.abs().amax(dim=0).clamp_min(1e-6),max=1.)
                policy.policy.action_net.weight.mul_(gain[:,None]);policy.policy.action_net.bias.mul_(gain)
                after=policy.policy.get_distribution(tensor).distribution.mean
            for parameter in policy.policy.action_net.parameters():policy.policy.optimizer.state.pop(parameter,None)
            calibration={'gain':gain.cpu().tolist(),'means_before':before.cpu().tolist(),'means_after':after.cpu().tolist()}
        if args.learning_rate is not None:
            policy.learning_rate=args.learning_rate
            policy.lr_schedule=lambda _: args.learning_rate
        if args.entropy is not None:policy.ent_coef=args.entropy
        if args.exploration_std is not None:
            if args.exploration_std<=0:raise ValueError('Exploration std must be positive')
            with torch.no_grad():policy.policy.log_std.fill_(np.log(args.exploration_std))
            policy.policy.optimizer.state.pop(policy.policy.log_std,None)
        if args.reset_critic:
            value_modules=(policy.policy.mlp_extractor.value_net,policy.policy.value_net)
            for module in value_modules:
                for layer in module.modules():
                    if isinstance(layer,torch.nn.Linear):
                        torch.nn.init.orthogonal_(layer.weight,gain=np.sqrt(2.) if module is value_modules[0] else 1.)
                        torch.nn.init.zeros_(layer.bias)
                for parameter in module.parameters():policy.policy.optimizer.state.pop(parameter,None)
        actor_after=actor_digest()
        if actor_before!=actor_after and not args.desaturate_actor:raise RuntimeError('Adaptation unexpectedly changed actor weights')
        features_preserved=all(torch.equal(v,features_before[k]) for k,v in policy.policy.mlp_extractor.policy_net.state_dict().items())
        if not features_preserved:raise RuntimeError('Actor feature network changed during adaptation setup')
        (root/'adaptation.json').write_text(json.dumps({'actor_preserved':actor_before==actor_after,
            'actor_features_preserved':features_preserved,'output_calibration':calibration,
            'actor_sha256':actor_after,'critic_reset':args.reset_critic,'variant':args.variant,
            'learning_rate':float(policy.lr_schedule(1.)), 'entropy':policy.ent_coef,
            'action_std':policy.policy.log_std.detach().exp().cpu().tolist()},indent=2))
        expected=torch.device(args.device)
        if next(policy.policy.parameters()).device.type!=expected.type:raise RuntimeError('Policy device mismatch')
        (root/'device.json').write_text(json.dumps({'requested':args.device,'actual':str(next(policy.policy.parameters()).device),
            'gpu_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'physics':'CPU MuJoCo',
            'workers':workers,'resumed_from':str(source) if source else None,'starting_timesteps':policy.num_timesteps},indent=2))
        print('RUN '+str(root),flush=True)
        checkpoint=CheckpointCallback(save_freq=max(1,args.checkpoint_every//workers),save_path=str(root/'checkpoints'),save_vecnormalize=True)
        status=RunStatus(root);callback=CallbackList([checkpoint,status])
        started=time.perf_counter()
        try:policy.learn(args.steps,callback=callback,reset_num_timesteps=not bool(source))
        finally:
            policy.save(str(root/'policy'));vector.save(str(root/'normalize.pkl'))
            status.write_status()
            (root/'training_summary.json').write_text(json.dumps({'timesteps':policy.num_timesteps,'elapsed_s':time.perf_counter()-started,'bullseye_claimed':False},indent=2))
    vector.close();print('OUTPUT '+str(root),flush=True)


if __name__=='__main__':main()
