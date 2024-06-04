from gymnasium.envs.registration import register


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"BinPick{suffix}-v2",
            entry_point="Abb_robot.env.fetch.bin_pick:MujocoBinPickEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )


__version__ = "1.2.3"

