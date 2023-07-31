from env import AutonomousDriving

if __name__ == '__main__':
    env = AutonomousDriving(
        map="celeste8P",
        n_agents=1,
        fixed_spawn=True,
        keepHistory=True,
    )

    number_of_episodes = 5
    number_of_steps = 50

    for i in range(number_of_episodes):
        obs = env.reset()
        for j in range(number_of_steps):
            env.render()

            # Select action
            action = env.action_space.sample() # Actions are integers from 0 to 8, see Car class in entities.py

            obs, reward, done, info = env.step(action, verbose=0)

            if done:
                break
        print("Episode {} finished".format(i))

    env.export_history("output")
