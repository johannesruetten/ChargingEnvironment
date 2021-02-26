from gym.envs.registration import register

register(id='ChargingEnv-v0',
    entry_point='envs.custom_env_dir:ChargingEnv'
)
