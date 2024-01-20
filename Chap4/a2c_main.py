# A2C main
# coded by St.Watermelon

## 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
from a2c_learn import A2Cagent
import gym

import os.path

def main():

    max_episode_num = 1000   # 최대 에피소드 설정
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    #env = gym.make(env_name, render_mode="human")  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    agent = A2Cagent(env)   # A2C 에이전트 객체

    # Load NN parameters
    if os.path.isfile('./save_weights/pendulum_actor.h5'):
        agent.load_weights('./save_weights/')  # 신경망 파라미터를 가져옴
        print("agent NN prams : load complete")


    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    main()