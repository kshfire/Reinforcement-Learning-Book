# A2C learn by Pytorch

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import TensorDataset, DataLoader


# A2C actor NN
class Actor(nn.Module):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        # cuda 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.h1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.h2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.h3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(16, action_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(16, action_dim),
            nn.Softplus()
        )

    def forward(self, state):
        # x: state
        out = self.h1(state)
        out = self.h2(out)
        out = self.h3(out)
        mu = self.mu(out)
        std = self.std(out)

        # 평균값을 [-action_bound, action_bound] 범위로 조정
        """
        mu = list(map(lambda x: x*self.action_bound, mu))
        mu = torch.FloatTensor([mu]).to(self.device)
        """
        mu = mu * self.action_bound

        #return [torch.transpose(mu, 0, 1), std]
        return [mu, std]

# A2C critic NN
class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.h2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.h3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Linear(16, 1)
        )
    def forward(self, state):
        out = self.h1(state)
        out = self.h2(out)
        out = self.h3(out)
        v = self.v(out)
        return v

# A2C Agent class
class A2CAgent(object):
    def __init__(self, env):
        # hyper parameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]
        # 표준편차의 최소값과 최대값 설정
        self.std_bound = [1e-2, 1.0]

        # cuda 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor 신경망 및 critic 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound).to(self.device)
        self.critic = Critic().to(self.device)

        print(self.actor)
        print(self.critic)

        print(f"device: {self.device}")


        # 옵티마이저 설정
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []


    # 로그 정책 확률밀도함수
    def log_pdf(self, mu, std, action):
        #std_array = std.cpu().detach().numpy()
        #std = np.clip(std_array, self.std_bound[0], self.std_bound[1])
        std = torch.clip(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * torch.log(var*2*np.pi)
        #log_policy_pdf = -0.5 * (action - mu.cpu().detach().numpy()) ** 2 / var - 0.5 * np.log(var * 2 * np.pi)
        return torch.sum(log_policy_pdf, 1, keepdim=True)
        #return np.sum(log_policy_pdf, 1, keepdims=True)


    # 액터 신경망에서 행동 샘플링
    def get_action(self, state):
        with torch.no_grad():
            mu_a, std_a = self.actor(state)

        std_a = torch.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = torch.normal(mu_a.item(), std_a.item(), size=(self.action_dim,))

        return action

    ## 액터 신경망 학습
    def actor_learn(self, states, actions, advantages):
        self.actor_opt.zero_grad()
        # 정책 확률밀도함수
        mu_a, std_a = self.actor(states)
        log_policy_pdf = self.log_pdf(mu_a, std_a, actions)
        # 손실함수
        loss_policy = log_policy_pdf * advantages
        loss = torch.sum(-loss_policy)
        # 그래디언트
        loss.backward()
        self.actor_opt.step()


    # critic 신경망 학습
    def critic_learn(self, states, td_targets):
        # self.critic_learn(torch.FloatTensor(states), torch.FloatTensor(td_targets))
        self.critic_opt.zero_grad()
        td_hat = self.critic(states)
        loss = F.mse_loss(td_targets, td_hat)
        loss.backward()
        self.critic_opt.step()


    # 시간차 타겟 계산
    def td_target(self, rewards, next_v_values, dones):
        #y_i = np.zeros(next_v_values.shape)
        y_i = torch.zeros(next_v_values.shape).to(self.device)
        for i in range(next_v_values.shape[0]):
            if dones[i]:
                y_i[i] = rewards[i]
            else:
                y_i[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return y_i

    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path + 'pendulum_actor.pt'))
        self.critic.load_state_dict(torch.load(path + 'pendulum_critic.pt'))


    # 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack

    ## Agent 학습
    def train(self, max_episode_num):
        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):
            # 배치 초기화
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()[0]

            while not done:
                # 학습 가시화
                # self.env.render()  # 이거 말고, gym 만들때 human_mode 로 해야 함

                # 행동 샘플링
                #action = self.get_action(torch.FloatTensor([state]).to(self.device))
                action = self.get_action(torch.FloatTensor(state).to(self.device))
                # 행동 범위 클리핑
                action = torch.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, trunc, _ = self.env.step(action)

                #print(f"action: {action}, next_state: {next_state}, reward: {reward}, done: {done}")

                if done or trunc:
                    done = True

                # shape 변환
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                done = np.reshape(done, [1, 1])
                # 학습용 보상 계산
                train_reward = (reward + 8) / 8.0

                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                # 배치가 채워질 때까지 학습하지 않고, 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                # 배치가 채워지면 학습 진행
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # 배치 비움
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

                # 시간차 타깃 계산
                with torch.no_grad():
                    next_v_values = self.critic(torch.FloatTensor(next_states).to(self.device))
                td_targets = self.td_target(torch.FloatTensor(train_rewards).to(self.device), next_v_values, dones)

                # 크리틱 신경망 업데이트
                self.critic_learn(torch.FloatTensor(states).to(self.device), td_targets)

                # 어드밴티지 계산
                with torch.no_grad():
                    v_values = self.critic(torch.FloatTensor(states).to(self.device))
                    next_v_values = self.critic(torch.FloatTensor(next_states).to(self.device))
                advantages = torch.FloatTensor(train_rewards).to(self.device) + self.GAMMA * next_v_values - v_values

                # actor 신경망 업데이트
                self.actor_learn(torch.FloatTensor(states).to(self.device),
                                 torch.FloatTensor(actions).to(self.device),
                                 advantages)

                # 상태 업데이트
                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            # 에피소드 결과 출력
            print(f"Episode: {ep+1}, Time: {time}, Reward: {episode_reward}")

            self.save_epi_reward.append(episode_reward)


            # 에피소드 10번마다 신경망 파라메터를 파일에 저장
            if ep % 10 == 0:
                torch.save(self.actor.state_dict(), "./save_weights/pendulum_actor.pt")
                torch.save(self.critic.state_dict(), "./save_weights/pendulum_critic.pt")


        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

