from gymnasium.envs.registration import register

register(
    id="PandaPickAndPlace-v0",
    entry_point="environments.panda_env:PandaPickAndPlaceEnv",
    max_episode_steps=150,
)
