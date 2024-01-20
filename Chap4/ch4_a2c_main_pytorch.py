

from ch4_a2c_learn_pytorch import A2CAgent
import gym

import os.path

def main():
    max_episode_num = 1000      # 최대 에피소드 설정

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)    # 환경으로 OpenAI Gym 의 pendulum-v0 설정
    agent = A2CAgent(env)       # A2C Agent 객체

    # Load NN parameters
    if os.path.isfile('./save_weights/pendulum_actor.pt'):
        agent.load_weights('./save_weights/')  # 신경망 파라미터를 가져옴
        print("agent NN prams : load complete")

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__ == "__main__":
    main()