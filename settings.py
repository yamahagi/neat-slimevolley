NODE_NAMES = {
    0: "forward",
    1: "backward",
    2: "jump",
    -1: "x",
    -2: "y",
    -3: "ball_relative_x",
    -4: "ball_relative_y",
    -5: "bvx",
    -6: "bvy",
}

TRAIN_MAX_STEPS = 3000
TEST_MAX_STEPS = 15000
GENERATIONS = 150
INTERVAL = 20

ATTENUATION = 0.95

HIT_COEF = 0.2
DIRECTION_COEF = 0.1
