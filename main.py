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
from settings import TRAIN_MAX_STEPS, TEST_MAX_STEPS, NODE_NAMES, GENERATIONS, INTERVAL


train_task = SlimeVolley(test=False, max_steps=TRAIN_MAX_STEPS, is_left_baseline=True)
test_task = SlimeVolley(test=True, max_steps=TEST_MAX_STEPS, is_left_baseline=True)
task_reset_fn = jax.jit(train_task.reset)
test_task_reset_fn = jax.jit(test_task.reset)
step_with_baseline_fn = jax.jit(train_task.step)
test_step_fn = jax.jit(test_task.step)

key = jax.random.PRNGKey(3)
generation = 0


def make_input(obs):
    x, y, _, _, bx, by, bvx, bvy, _, _, _, _ = obs
    return np.array([x, y, bx - x, by - y, bvx, bvy])


def game_with_baseline(net):
    global key
    global generation
    key, subkey = jax.random.split(key)
    task_state = task_reset_fn(subkey[None, :], jnp.array([generation]))
    score_sum = 0
    weighted_score_sum_right = 0
    while True:
        obs_right = task_state.obs[1].flatten()
        action = jnp.array(net.activate(make_input(obs_right))).reshape(1, -1)
        task_state, reward, _, weighted_score_right, done = step_with_baseline_fn(
            task_state, action, action, jnp.array([generation])
        )
        score_sum += reward[0].item()
        weighted_score_sum_right += weighted_score_right[0].item()
        if done:
            break
    return score_sum, weighted_score_sum_right


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


def eval_genomes(genomes, config):
    global generation
    nets = [
        neat.nn.FeedForwardNetwork.create(genome, config)
        for genome_id, genome in genomes
    ]

    for net, genome in zip(nets, genomes):
        genome[1].fitness = 0.0
    for net, genome in zip(nets, genomes):
        for _ in range(3):
            genome[1].fitness += game_with_baseline(net)[1]

    if generation % INTERVAL == 0:
        best_net_id = np.argmax([genome[1].fitness for genome in genomes])
        best_net = neat.nn.FeedForwardNetwork.create(genomes[best_net_id][1], config)
        with open(log_dir / Path(f"{generation}_best.pkl"), "wb") as fw:
            pickle.dump(genomes[best_net_id][1], fw)
        visualize.draw_net(
            config,
            genomes[best_net_id][1],
            True,
            node_names=NODE_NAMES,
            filename=log_dir / Path(f"generation_{generation}"),
        )
        test(best_net, log_dir / Path(f"{generation}_best.gif"))
    generation += 1


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(INTERVAL, filename_prefix=log_dir / Path("checkpoint"))
    )

    winner = p.run(eval_genomes, GENERATIONS)

    with open(log_dir / Path("best.pkl"), "wb") as f:
        pickle.dump(winner, f)

    visualize.draw_net(
        config, winner, True, node_names=NODE_NAMES, filename=log_dir / Path(f"bestnet")
    )
    with open(log_dir / Path(f"stats.pkl"), "wb") as fw:
        pickle.dump(stats, fw)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    final_score = test(winner_net, log_dir / Path("test.gif"))
    print(final_score)


if __name__ == "__main__":
    args = sys.argv
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    log_dir = Path("./logs/" + str(args[1]))
    log_dir.mkdir(parents=True, exist_ok=True)
    run(config_path)
