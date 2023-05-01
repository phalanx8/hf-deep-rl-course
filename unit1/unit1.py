from pyvirtualdisplay import Display
from huggingface_hub import login
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


if __name__ == "main":
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    login()

    env_id = "LunarLander-v2"
    env = make_vec_env(env_id, n_envs=16)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1)

    # Train it for 1,000,000 timesteps
    model.learn(total_timesteps=100000000)
    # Save the model
    model_name = "mlp-ppo-gym-LunarLander-v2"
    model.save(model_name)

    eval_env = gym.make("LunarLander-v2")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    model_architecture = "PPO"
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    package_to_hub(model=model, model_id=model_name, model_type=model_architecture, env=eval_env, training_env=env)
