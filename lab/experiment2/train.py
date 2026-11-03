"""Train/resume SB3 PPO and preserve every attempted throw as states + video."""
import runtime_bootstrap  # before Pandas/SB3 in both parent and spawned workers
import argparse
import sys
import os
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
from env import DartThrowEnv,VARIANTS,build_env
from bounded_policy import BoundedActorCriticPolicy


def normalizer_for(model):
    model=Path(model)
    if model.name.startswith('rl_model_'):
        return model.with_name('rl_model_vecnormalize_'+model.stem.removeprefix('rl_model_')+'.pkl')
    return model.parent/'normalize.pkl'


class RunStatus(BaseCallback):
    def __init__(self,root):
        super().__init__();self.root=root;self.recent=deque(maxlen=100);self.completed=0
        self.action_elements=0;self.clipped_elements=0
    def write_status(self):
        policy_device=str(next(self.model.policy.parameters()).device)
        optimizer_devices=sorted({str(v.device) for state in self.model.policy.optimizer.state.values() for v in state.values() if torch.is_tensor(v) and v.ndim>0})
        status={'timesteps':self.num_timesteps,'policy_device':policy_device,'optimizer_tensor_devices':optimizer_devices,
                'workers':self.training_env.num_envs,'physics_device':'cpu','updated_utc':datetime.now(timezone.utc).isoformat(),
                'action_distribution':type(self.model.policy.action_dist).__name__,
                'action_std':self.model.policy.log_std.detach().exp().cpu().tolist(),
                'external_action_clip_fraction':self.clipped_elements/self.action_elements if self.action_elements else None}
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
            from throw_diagnostics import diagnostics
            metrics.update(diagnostics(rows))
            for name in ('parameter_reward','release_delay_s'):
                values=[r[name] for r in rows if name in r]
                if values:metrics[name+'_mean']=float(np.mean(values))
            parameters=np.array([r['throw_parameters'] for r in rows if 'throw_parameters' in r])
            if parameters.size:
                metrics['parameter_mean']=parameters.mean(axis=0).tolist()
                metrics['parameter_std']=parameters.std(axis=0).tolist()
                metrics['parameter_boundary_fraction']=(np.abs(parameters)>.95).mean(axis=0).tolist()
            temp=self.root/'progress_metrics.tmp';temp.write_text(json.dumps(metrics,indent=2));temp.replace(self.root/'progress_metrics.json')
            for key in ('median_velocity_gap_m_s','median_vertical_deficit_m_s','median_predicted_board_error_m'):
                if metrics[key] is not None:self.logger.record('throw/'+key,metrics[key])
    def _on_training_start(self):self.write_status()
    def _on_rollout_start(self):self.write_status()
    def _on_step(self):
        actions=np.asarray(self.locals.get('actions',()))
        self.action_elements+=actions.size;self.clipped_elements+=int(np.count_nonzero(np.abs(actions)>1))
        for done,info in zip(self.locals.get('dones',()),self.locals.get('infos',())):
            if done:self.recent.append(info);self.completed+=1
        return not (self.root/'STOP').exists()


class Archive(gym.Wrapper):
    def __init__(self,env,folder):
        super().__init__(env);self.folder=Path(folder);self.number=0;self.actions=[]
        self.folder.mkdir(parents=True,exist_ok=True)
    def reset(self,**kwargs):
        self.actions=[]
        return self.env.reset(**kwargs)
    def step(self,action):
        start=self.env.t
        result=self.env.step(action)
        if hasattr(self.env,'transition_controls'):
            self.actions.extend(self.env.transition_controls)
        else:
            self.actions.append((start,np.asarray(action).copy(),float(result[1]),
                self.env.sim.target.copy(),self.env.sim.velocity_target.copy(),self.env.d.ctrl.copy()))
        if result[2] or result[3]:self.save(result[4])
        return result
    def save(self,info):
        s=self.env;folder=self.folder/f'{self.number:06d}';self.number+=1
        folder.mkdir(exist_ok=False)
        (folder/'result.json').write_text(json.dumps(info,indent=2))
        np.savez_compressed(folder/'states.npz',time=[v[0] for v in s.frames],qpos=[v[1] for v in s.frames],qvel=[v[2] for v in s.frames])
        np.savez_compressed(folder/'actions.npz',time=[v[0] for v in self.actions],action=[v[1] for v in self.actions],
            reward=[v[2] for v in self.actions],joint_target=[v[3] for v in self.actions],
            velocity_target=[v[4] for v in self.actions],actuator_torque=[v[5] for v in self.actions])
        # Render archived states, restoring the terminal state for Gym/Monitor.
        final_q=s.d.qpos.copy();final_v=s.d.qvel.copy()
        pipe=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','800x480','-r','50','-i','-',
                               '-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'swing.mp4')],stdin=subprocess.PIPE)
        try:
            for _,q,v in s.frames:
                s.d.qpos[:]=q;s.d.qvel[:]=v;mujoco.mj_forward(s.m,s.d)
                pipe.stdin.write(s.render().tobytes())
            pipe.stdin.close()
            if pipe.wait():raise RuntimeError('Video encoding failed')
        finally:
            if pipe.poll() is None:pipe.terminate();pipe.wait()
            s.d.qpos[:]=final_q;s.d.qvel[:]=final_v;mujoco.mj_forward(s.m,s.d)


class RewardScale(gym.RewardWrapper):
    """Scale PPO returns only; Archive retains the original physical rewards."""
    def __init__(self,env,scale):
        super().__init__(env);self.scale=float(scale)
    def reward(self,reward):return self.scale*reward


def rescale_value_estimator(policy,factor):
    """Keep the actor unchanged and convert the critic to new reward units."""
    if factor==1.:return
    with torch.no_grad():
        policy.policy.value_net.weight.mul_(factor)
        policy.policy.value_net.bias.mul_(factor)
    for module in (policy.policy.mlp_extractor.value_net,policy.policy.value_net):
        for parameter in module.parameters():policy.policy.optimizer.state.pop(parameter,None)


def factory(folder,rank,randomize,variant='legacy',throw_profile=None,reward_scale=1.):
    def create():
        raw=Archive(build_env(randomize=randomize,record=True,variant=variant,throw_profile=throw_profile),Path(folder)/'episodes'/f'worker_{rank}')
        return Monitor(RewardScale(raw,reward_scale),str(Path(folder)/f'monitor_{rank}.csv'))
    return create


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1_000_000)
    p.add_argument('--run-root',type=Path,help='Optional experiment-specific run archive')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--rollout',type=int,default=512);p.add_argument('--randomize',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--evaluate',type=Path);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    p.add_argument('--checkpoint-every',type=int,default=2048)
    p.add_argument('--variant',choices=VARIANTS,default=None)
    p.add_argument('--throw-profile',type=Path,help='Joint-reference calibration JSON; inherited from checkpoint when omitted')
    p.add_argument('--freeze-normalization',action=argparse.BooleanOptionalAction,default=None,
                   help='Keep paired observation statistics fixed while PPO updates; inherited on resume')
    p.add_argument('--reset-critic',action='store_true',help='Reinitialize only the value estimator after changing rewards')
    p.add_argument('--learning-rate',type=float,default=None)
    p.add_argument('--anchor-checkpoint',type=Path,default=None)
    p.add_argument('--anchor-coef',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--target-kl',type=float,default=None)
    p.add_argument('--clip-range',type=float,default=None)
    p.add_argument('--reward-scale',type=float,default=None,help='Positive constant PPO reward scale; inherited on resume, critic units converted automatically')
    p.add_argument('--entropy',type=float,default=None)
    p.add_argument('--exploration-std',type=float,default=None,help='Explicit new action std when adapting to changed rewards')
    p.add_argument('--desaturate-actor',action='store_true',help='Calibrate saturated mean-action output rows to <=0.8 at reset; retain actor features')
    p.add_argument('--bounded-actions',action='store_true',help='Explicitly migrate/create a tanh-Gaussian policy with corrected log probabilities')
    p.add_argument('--regression-guard',action='store_true',help='Run independent physical-regression stop monitor')
    p.add_argument('--eval-every',type=int,default=0,help='Optional periodic deterministic physical validation interval')
    p.add_argument('--eval-episodes',type=int,default=8);p.add_argument('--eval-seed',type=int,default=76101)
    p.add_argument('--reference-checkpoint',type=Path,help='Preserve an independently evaluated fallback in the physical best selection')
    args=p.parse_args();torch.set_num_threads(1)
    source=args.evaluate or args.resume
    config_dir=source.parent.parent if source and source.parent.name=='checkpoints' else (source.parent if source else None)
    config=json.loads((config_dir/'config.json').read_text()) if config_dir and (config_dir/'config.json').exists() else {}
    if args.variant is None:
        args.variant=config.get('variant') or 'legacy'
    profile=json.loads(args.throw_profile.read_text()) if args.throw_profile else config.get('throw_profile')
    if profile is not None and 'profile' in profile:profile=profile['profile']
    args.throw_profile=profile
    if args.anchor_checkpoint is None and config.get('anchor_checkpoint'):
        args.anchor_checkpoint=Path(config['anchor_checkpoint'])
        args.anchor_coef=config.get('anchor_coef',1.)
    source_reward_scale=float(config.get('reward_scale',1.))
    if args.reward_scale is None:args.reward_scale=source_reward_scale
    if not np.isfinite(args.reward_scale) or args.reward_scale<=0:raise ValueError('Reward scale must be positive and finite')
    if args.freeze_normalization is None:args.freeze_normalization=config.get('freeze_normalization',False)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    if min(args.steps,args.workers,args.rollout,args.episodes)<1:raise ValueError('Counts must be positive')
    if args.eval_every<0 or args.eval_episodes<1:raise ValueError('Invalid physical evaluation counts')
    if args.rollout*args.workers<2:raise ValueError('PPO needs at least two rollout samples')
    root=(args.run_root or Path(__file__).parent/'runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    root.mkdir(parents=True);(root/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    source_dir=Path(__file__).parent;repo=source_dir.parent;manifest={}
    for name in ('experiment2/env.py','experiment2/train.py','experiment2/trajectory.py','experiment2/bounded_policy.py','experiment2/anchored_ppo.py','experiment2/parameter_env.py','experiment2/throw_diagnostics.py','experiment2/remote_training/guard.py','experiment2/physics_eval.py','experiment2/motion_prior.py','experiment2/calibrated_env.py','experiment2/runtime_bootstrap.py','experiment1_throw.py','inspire_grip.py','inspire_model.py','inspire_grip/policy.json','dart_model.py','dart_profile.json'):
        path=repo/name;manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    video_profile=(args.throw_profile or {}).get('controller')=='video_to_throw_v1'
    if video_profile:
        for name in ('experiment4/video_env.py','experiment4/train_video.py','experiment4/preflight.py',args.throw_profile['reference']):
            path=repo/name;manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    if (args.throw_profile or {}).get('controller')=='parameter_ppo_v1':
        teacher=Path(args.throw_profile['teacher_checkpoint'])
        if not teacher.is_absolute():teacher=repo/teacher
        manifest['frozen_teacher_policy']=hashlib.sha256(teacher.read_bytes()).hexdigest()
        manifest['frozen_teacher_normalizer']=hashlib.sha256((teacher.parent/'normalize.pkl').read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    workers=1 if args.evaluate else args.workers
    vector=(DummyVecEnv if workers==1 else SubprocVecEnv)([factory(root,i,args.randomize,args.variant,args.throw_profile,args.reward_scale) for i in range(workers)])
    if source:
        vector=VecNormalize.load(str(normalizer_for(source)),vector)
    else:vector=VecNormalize(vector,norm_obs=True,norm_reward=False,clip_obs=10.)
    vector.seed(args.seed)
    if video_profile and not source:
        # The new motion has different reset observations. Calibrate its own
        # statistics before freezing them, without taking optimization steps.
        for _ in range(8):vector.reset()
        vector.save(str(root/'initial_normalize.pkl'))
    if args.freeze_normalization:vector.training=False
    load_options={'custom_objects':{'policy_class':BoundedActorCriticPolicy}} if args.bounded_actions else {}
    if args.evaluate:
        vector.training=False;policy=PPO.load(str(source),env=vector,device=args.device,seed=args.seed,**load_options)
        obs=vector.reset();reports=[]
        while len(reports)<args.episodes:
            action,_=policy.predict(obs,deterministic=True);obs,rewards,dones,info=vector.step(action)
            if dones[0]:reports.append(info[0])
        # Terminal observations are numpy arrays; keep outcome metrics only.
        reports=[{k:v for k,v in r.items() if k not in ('terminal_observation','episode')} for r in reports]
        (root/'evaluation.json').write_text(json.dumps({'seed':args.seed,'episodes':reports,'bullseye_rate':sum(r['is_success'] for r in reports)/len(reports)},indent=2))
    else:
        from anchored_ppo import AnchoredPPO
        trainer=AnchoredPPO if args.anchor_checkpoint else PPO
        policy=trainer.load(str(source),env=vector,device=args.device,seed=args.seed,n_steps=args.rollout,batch_size=min(256,args.rollout*workers),**load_options) if source else PPO(BoundedActorCriticPolicy if args.bounded_actions else 'MlpPolicy',vector,device=args.device,seed=args.seed,
            n_steps=args.rollout,batch_size=min(64,args.rollout*workers),learning_rate=3e-4,n_epochs=10,
            gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.005,target_kl=.03,
            policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128]),'log_std_init':-.7},verbose=1)
        if video_profile and not source:
            with torch.no_grad():
                policy.policy.action_net.weight.zero_()
                policy.policy.action_net.bias.zero_()
        def actor_digest():
            digest=hashlib.sha256()
            for module in (policy.policy.mlp_extractor.policy_net,policy.policy.action_net):
                for parameter in module.parameters():digest.update(parameter.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        if args.anchor_checkpoint:
            if not args.freeze_normalization:raise ValueError('Anchored training requires frozen observation statistics')
            with normalizer_for(args.anchor_checkpoint).open('rb') as handle:
                import cloudpickle
                reference_norm=cloudpickle.load(handle)
            if not np.array_equal(reference_norm.obs_rms.mean,vector.obs_rms.mean) or not np.array_equal(reference_norm.obs_rms.var,vector.obs_rms.var):
                raise ValueError('Anchor and learner must use identical observation normalization')
            policy.set_anchor(args.anchor_checkpoint,args.anchor_coef)
        actor_before=actor_digest()
        if args.batch_size is not None:
            if args.batch_size<2 or (args.rollout*workers)%args.batch_size:raise ValueError('Batch size must divide rollout buffer')
            policy.batch_size=args.batch_size
        if args.epochs is not None:
            if args.epochs<1:raise ValueError('Epochs must be positive')
            policy.n_epochs=args.epochs
        if args.target_kl is not None:
            if args.target_kl<=0:raise ValueError('Target KL must be positive')
            policy.target_kl=args.target_kl
        if args.clip_range is not None:
            if not 0<args.clip_range<1:raise ValueError('Clip range must be between zero and one')
            from stable_baselines3.common.utils import FloatSchedule
            policy.clip_range=FloatSchedule(args.clip_range)
        if source:rescale_value_estimator(policy,args.reward_scale/source_reward_scale)
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
            from stable_baselines3.common.utils import FloatSchedule
            policy.lr_schedule=FloatSchedule(args.learning_rate)
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
            'policy_class':type(policy.policy).__name__,'action_distribution':type(policy.policy.action_dist).__name__,
            'actor_features_preserved':features_preserved,'output_calibration':calibration,
            'actor_sha256':actor_after,'critic_reset':args.reset_critic,'variant':args.variant,
            'reward_scale':args.reward_scale,'source_reward_scale':source_reward_scale,
            'learning_rate':float(policy.lr_schedule(1.)), 'entropy':policy.ent_coef,
            'action_std':policy.policy.log_std.detach().exp().cpu().tolist()},indent=2))
        expected=torch.device(args.device)
        if next(policy.policy.parameters()).device.type!=expected.type:raise RuntimeError('Policy device mismatch')
        (root/'device.json').write_text(json.dumps({'requested':args.device,'actual':str(next(policy.policy.parameters()).device),
            'gpu_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'physics':'CPU MuJoCo',
            'workers':workers,'resumed_from':str(source) if source else None,'starting_timesteps':policy.num_timesteps},indent=2))
        from stable_baselines3.common.logger import configure
        policy.set_logger(configure(str(root/'logs'),['stdout','csv']))
        print('RUN '+str(root),flush=True)
        if video_profile:
            status_path=repo/'experiment4/status.json'
            metadata=json.loads(status_path.read_text()) if status_path.exists() else {}
            metadata.update(name='Video to Throw PPO',state='running',pid=os.getpid(),run_dir=str(root),
                            guard_status=str(root/'guard.json'),workers=workers,rollout=args.rollout,
                            starting_steps=policy.num_timesteps,additional_steps=args.steps,
                            step_unit='complete physical throws',location='ssh',revision='video_to_throw_v1')
            for path in (status_path,repo/'experiment2/remote_training/status.json'):
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(metadata,indent=2));temporary.replace(path)
        if args.regression_guard:
            if not args.eval_every:raise ValueError('Regression guard requires physical evaluations')
            with (root/'guard.log').open('w') as guard_log:
                subprocess.Popen([sys.executable,str(Path(__file__).parent/'remote_training/guard.py'),
                    '--run',str(root),'--pid',str(os.getpid())],stdout=guard_log,stderr=guard_log)

        checkpoint=CheckpointCallback(save_freq=max(1,args.checkpoint_every//workers),save_path=str(root/'checkpoints'),save_vecnormalize=True)
        (root/'checkpoints').mkdir(exist_ok=True)
        shutil.copy2(root/'config.json',root/'checkpoints/config.json')
        status=RunStatus(root);callbacks=[checkpoint,status]
        if args.eval_every:
            from physics_eval import PhysicsEvaluation
            reference=None
            if args.reference_checkpoint:
                ref=args.reference_checkpoint;refroot=ref.parent.parent if ref.parent.name=='checkpoints' else ref.parent
                reference=(ref,normalizer_for(ref),refroot/'config.json')
            callbacks.append(PhysicsEvaluation(root,args.variant,args.randomize,args.eval_every,args.eval_episodes,args.eval_seed,Archive,reference))
        callback=CallbackList(callbacks)
        started=time.perf_counter()
        failure=None
        try:policy.learn(args.steps,callback=callback,reset_num_timesteps=not bool(source))
        except BaseException as error:
            failure=type(error).__name__+': '+str(error)
            raise
        finally:
            policy.save(str(root/'policy'));vector.save(str(root/'normalize.pkl'))
            status.write_status()
            (root/'training_summary.json').write_text(json.dumps({'timesteps':policy.num_timesteps,'elapsed_s':time.perf_counter()-started,'bullseye_claimed':False,'status':'failed' if failure else ('stopped' if (root/'STOP').exists() else 'completed'),'error':failure},indent=2))
    vector.close();print('OUTPUT '+str(root),flush=True)


if __name__=='__main__':main()
