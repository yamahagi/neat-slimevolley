import jax
import jax.numpy as jnp
import numpy as np
import neat
import visualize
import os
import pickle
import sys
from pathlib import Path

from slimevolley import SlimeVolley
from settings import TEST_MAX_STEPS


test_task = SlimeVolley(test=True, max_steps=TEST_MAX_STEPS, is_left_baseline=True)
test_task_reset_fn = jax.jit(test_task.reset)
test_step_fn = jax.jit(test_task.step)

key = jax.random.PRNGKey(3)
generation = 0


def make_input(obs):
    x, y, _, _, bx, by, bvx, bvy, _, _, _, _ = obs
    return np.array([x, y, bx - x, by - y, bvx, bvy])

def test(best_net, save_file):
    global key
    global generation
    task_state = test_task_reset_fn(key[None, :], jnp.array([generation]))
    images = []
    reward_all = 0
    for _ in range(TEST_MAX_STEPS):
        obs_right = task_state.obs[1]
        obs = obs_right.flatten()
        action = jnp.array(best_net.activate(make_input(obs)))
        action = action.reshape(1, -1)
        task_state, reward, _, _, done = test_step_fn(
            task_state, action, action, jnp.array([generation])
        )
        reward_all += reward
        images.append(SlimeVolley.render(task_state))
        if done:
            print("done")
            break

    images[0].save(
        save_file, save_all=True, append_images=images[1:], duration=20, loop=0
    )

    return reward_all


def run(config_file, final_model_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    with open(final_model_path, "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    final_score = test(winner_net, log_dir / Path("test.gif"))
    print(final_score)


if __name__ == "__main__":
    args = sys.argv
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    log_dir = Path("./logs/" + str(args[1]))
    log_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = args[2]
    run(config_path, final_model_path)
