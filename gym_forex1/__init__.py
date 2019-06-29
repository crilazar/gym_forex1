from gym.envs.registration import register

register(
    id='forex1-v0',
    entry_point='gym_forex1.envs:Forex1',
)