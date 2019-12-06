import numpy as np
import tensorflow as tf

from maml_rl.envs.navigation import Navigation2DEnv
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.baselines.linear_baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.optimizers import ConjugateGradientOptimizer
from maml_rl.metalearners.maml_trpo import MetaLearner

ITR = 1

META_POLICY_PATH = "saves/maml/policy-{}".format(ITR)
BASELINE_PATH = "saves/maml/baseline-{}".format(ITR)

GOALS = np.array([0.3, 0.2])
TEST_TASKS = [{'goal': goal} for goal in GOALS]


def load_meta_learner_params(policy_path, baseline_path, env):

    with tf.name_scope('policy') as scope:
        policy = NormalMLPPolicy(
            int(np.prod(env.observation_space.shape)),
            int(np.prod(env.action_space.shape)),
            hidden_sizes=(100, 100),
            name=scope
        )  # We should actually get this from config
    print(policy.get_trainable_variables())
    policy.load_weights(policy_path)
    print("")
    print(policy.get_trainable_variables())

    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
    baseline.load_weights(baseline_path)

    return policy, baseline


def evaluate(env, task, policy, max_path_length=100):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    for _ in range(max_path_length):
        env.render()
        with tf.device('cpu:0'):
            obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32) # Device?
        action_tensor = policy(obs_tensor, params=None).sample()
        with tf.device('cpu:0'):
            action = action_tensor.numpy()
        obs, rew, done, _ = env.step(action)
        cum_reward += rew
        t += 1
        if done:
            break

    print("========EVAL RESULTS=======")
    print("Return: {}, Timesteps:{}".format(cum_reward, t))
    print("===========================")


def main():
    env = Navigation2DEnv()
    policy, baseline = load_meta_learner_params(META_POLICY_PATH, BASELINE_PATH, env)
    sampler = BatchSampler(env_name="2DNavigation-v0", batch_size=20, num_workers=2)
    learner = MetaLearner(sampler, policy, baseline, optimizer=None)

    for task in TEST_TASKS:
        env.reset_task(task)

        # Sample a batch of transitions
        sampler.reset_task(task)
        episodes = sampler.sample(policy)
        new_params = learner.adapt(episodes)
        policy.set_params_with_name(new_params)
        evaluate(env, task, policy)


if __name__ == '__main__':
    main()
