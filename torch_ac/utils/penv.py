import multiprocessing
import gymnasium as gym


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            # Gymnasium API
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                # Gymnasium reset returns (obs, info)
                obs, _ = env.reset()
            # Send old-style tuple (no terminated/truncated)
            conn.send((obs, reward, done, info))

        elif cmd == "reset":
            # Gymnasium API
            obs, _ = env.reset()
            # Send only obs to match old interface
            conn.send(obs)

        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        # Ask all remote envs to reset
        for local in self.locals:
            local.send(("reset", None))

        # Local env reset (Gymnasium: returns (obs, info))
        obs0, _ = self.envs[0].reset()

        # Remote envs send just obs (old-style)
        results = [obs0] + [local.recv() for local in self.locals]
        # Old interface: list of observations
        return results

    def step(self, actions):
        # Send step to all remote envs except the first
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))

        # Step first env locally (Gymnasium API)
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        done = terminated or truncated
        if done:
            obs, _ = self.envs[0].reset()

        # Collect all results as old-style tuples
        # First local, then remotes (which send (obs, reward, done, info))
        results = zip(
            *(
                [(obs, reward, done, info)]
                + [local.recv() for local in self.locals]
            )
        )
        # Old interface: returns an iterator of 4 sequences
        # so you can do: obs, reward, done, info = env.step(actions)
        return results

    def render(self):
        raise NotImplementedError
