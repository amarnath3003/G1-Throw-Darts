"""Periodic deterministic physics validation and best-checkpoint preservation."""
import json
from pathlib import Path
import shutil
import cloudpickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import build_env


def physical_summary(rows,start_distance):
    clean=[r.get('release') is not None and r['outcome'] in ('environment_hit','board','timeout') for r in rows]
    distances=[float(r['closest_tip_distance_m']) if good and r.get('closest_tip_distance_m') is not None else start_distance+1.
               for r,good in zip(rows,clean)]
    success=sum(bool(r.get('is_success')) for r in rows)
    board=sum(r.get('reward_terms',{}).get('board_impact',0)>0 for r in rows)
    rate=float(np.mean(clean));distance=float(np.mean(distances))
    return {'episodes':len(rows),'bullseyes':success,'valid_board_hits':board,'clean_flight_rate':rate,
            'mean_penalized_closest_distance_m':distance,
            'selection_score':[int(rate>=.8),success,board,-distance]}


class PhysicsEvaluation(BaseCallback):
    def __init__(self,root,variant,randomize,every,episodes,seed,archive_class,reference=None):
        super().__init__();self.root=Path(root);self.variant=variant;self.randomize=randomize
        self.every=every;self.episodes=episodes;self.seed=seed;self.archive_class=archive_class
        self.best=None;self.next_step=None
        self.reference=reference

    def evaluate(self,policy=None,normalizer=None,variant=None,config=None,label='',normalizer_source=None):
        if (self.root/'STOP').exists():return
        policy=self.model if policy is None else policy
        variant=self.variant if variant is None else variant
        folder=self.root/'evaluations'/(f'{self.num_timesteps:09d}'+label);folder.mkdir(parents=True,exist_ok=False)
        normalizer=self.model.get_vec_normalize_env() if normalizer is None else normalizer
        if normalizer is None:raise RuntimeError('Physical evaluation requires paired observation normalization')
        # Save the exact policy/statistics used in these synchronous rollouts.
        policy.save(str(folder/'policy'))
        # A reference normalizer is intentionally unbound to a training VecEnv.
        # Its original serialized bytes are the exact paired statistics; saving
        # that unbound object again assumes VecEnv-only attributes in SB3.
        if normalizer_source is not None:shutil.copy2(normalizer_source,folder/'normalize.pkl')
        else:normalizer.save(str(folder/'normalize.pkl'))
        config=json.loads((self.root/'config.json').read_text()) if config is None else config
        (folder/'config.json').write_text(json.dumps(config,indent=2))
        env=self.archive_class(build_env(variant=variant,throw_profile=config.get('throw_profile'),randomize=self.randomize,record=True),folder/'episodes')
        rows=[];distances=[];interrupted=False
        try:
            for seed in range(self.seed,self.seed+self.episodes):
                if (self.root/'STOP').exists():interrupted=True;break
                obs,_=env.reset(seed=seed);distances.append(float(np.linalg.norm(env.unwrapped.goal-env.unwrapped.start_pos)))
                done=False
                while not done:
                    if (self.root/'STOP').exists():interrupted=True;break
                    # normalize_obs does not update obs_rms; training workers
                    # are idle while this callback runs, so stats remain fixed.
                    action,_=policy.predict(normalizer.normalize_obs(obs[None,:]),deterministic=True)
                    obs,_,term,trunc,info=env.step(action[0]);done=term or trunc
                if interrupted:break
                rows.append(dict(info,seed=seed))
        finally:env.close()
        if interrupted:
            (folder/'evaluation.json').write_text(json.dumps({'interrupted':True,'timesteps':self.num_timesteps,
                'episodes':rows,'note':'Incomplete validation; excluded from best-checkpoint selection.'},indent=2))
            return
        summary=physical_summary(rows,float(np.mean(distances)))
        report={'timesteps':self.num_timesteps,'policy_timesteps':policy.num_timesteps,'seeds':list(range(self.seed,self.seed+self.episodes)),
                'variant':variant,'randomize':self.randomize,'summary':summary,'episodes':rows,'label':label}
        (folder/'evaluation.json').write_text(json.dumps(report,indent=2))
        score=tuple(summary['selection_score'])
        if self.best is None or score>self.best:
            self.best=score;best=self.root/'best_physics';best.mkdir(exist_ok=True)
            for name in ('policy.zip','normalize.pkl','config.json','evaluation.json'):shutil.copy2(folder/name,best/name)
            (self.root/'best_physics.json').write_text(json.dumps({'evaluation':str(folder),'summary':summary,
                'checkpoint':str(best/'policy.zip')},indent=2))
        for key in ('bullseyes','valid_board_hits','clean_flight_rate','mean_penalized_closest_distance_m'):
            self.logger.record('physics_eval/'+key,summary[key])
        print('PHYSICS_EVAL '+json.dumps({'timesteps':self.num_timesteps,'variant':variant,'label':label,**summary}),flush=True)

    def _on_training_start(self):
        if self.reference:
            model_path,norm_path,config_path=self.reference
            policy=PPO.load(str(model_path),device=self.model.device)
            with Path(norm_path).open('rb') as handle:normalizer=cloudpickle.load(handle)
            config=json.loads(Path(config_path).read_text())
            self.evaluate(policy,normalizer,config.get('variant') or 'legacy',config,label='_reference',normalizer_source=norm_path)
            del policy,normalizer
        self.evaluate();self.next_step=self.num_timesteps+self.every

    def _on_step(self):
        if (self.root/'STOP').exists():return False
        if self.num_timesteps>=self.next_step:
            self.evaluate();self.next_step=self.num_timesteps+self.every
        return True
