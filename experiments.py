import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import Monitor
import numpy.random as np_random
from mujoco_py import GlfwContext
#GlfwContext(offscreen=True)  # Create a window to init GLFW.

from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.baselines.linear_baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.metalearners.maml_trpo import MetaLearner

ITR = 991
GRAD_STEPS = [0, 1]
EVAL_STEPS = 10

ENV_NAME = "HalfCheetahDir-v1"
ENV_PREFIX = "halfcheetah-dir"

META_POLICY_PATH = f"saves/maml-{ENV_PREFIX}/policy-{ITR}"
BASELINE_PATH = f"saves/maml-{ENV_PREFIX}/baseline-{ITR}"

directions = 2 * np_random.binomial(1, p=0.5, size=(1,)) - 1
TEST_TASKS = [{'direction': direction} for direction in directions]

GOALS = np.array([[0.3, 0.2]])
#TEST_TASKS = [{'goal': goal} for goal in GOALS]


def load_meta_learner_params(policy_path, baseline_path, env):

    with tf.name_scope('policy') as scope:
        policy = NormalMLPPolicy(
            int(np.prod(env.observation_space.shape)),
            int(np.prod(env.action_space.shape)),
            hidden_sizes=(100, 100),
            name=scope
        )
    policy.load_weights(policy_path)

    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
    baseline.load_weights(baseline_path)

    return policy, baseline


def evaluate(env, task, policy):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    while True:
        env.render(mode='rgb_array')
        with tf.device('cpu:0'):
            obs_tensor = tf.convert_to_tensor(obs[np.newaxis], dtype=tf.float32)

        action_tensor = policy(obs_tensor, params=None).sample()

        with tf.device('cpu:0'):
            action = action_tensor.numpy()
        obs, rew, done, _ = env.step(action[0])
        cum_reward += rew
        t += 1
        if done:
            break

    return cum_reward


def main():
    env = gym.make(ENV_NAME)
    env = Monitor(env, f"./videos/{ENV_PREFIX}", force=True)
    policy, baseline = load_meta_learner_params(META_POLICY_PATH, BASELINE_PATH, env)
    sampler = BatchSampler(env_name=ENV_NAME, batch_size=20, num_workers=2)
    learner = MetaLearner(sampler, policy, baseline, optimizer=None)

    for task in TEST_TASKS:
        returns = []

        for i in range(1, EVAL_STEPS+1):
            for grad_steps in GRAD_STEPS:
                if i % 10 == 0:
                    print(f"Evaluation-step: {i}")

                env.reset_task(task)
                learner.policy, learner.baseline = load_meta_learner_params(META_POLICY_PATH,
                                                                            BASELINE_PATH,
                                                                            env)

                # Sample a batch of transitions
                sampler.reset_task(task)
                episodes = sampler.sample(learner.policy)
                for _ in range(grad_steps):
                    new_params = learner.adapt(episodes)
                    learner.policy.set_params_with_name(new_params)
                returns.append(evaluate(env, task, learner.policy))

        print("========EVAL RESULTS=======")
        print(f"Task: {task}")
        print(f"Returns: {returns}")
        print(f"Average Return: {np.mean(returns)}")
        print("===========================")


if __name__ == '__main__':
    main()
