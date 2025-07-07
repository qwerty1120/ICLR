import torch
from torch.nn import functional as F
import numpy as np
from utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'
# @@
def exponential_beta_schedule_discrete(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    Exponential Scheduler (Discrete Version)
    -----------------------------------------
    - timesteps: 전체 타임스텝 수 (최종 출력 β 배열의 길이는 timesteps+1)
    - beta_start, beta_end: 시작 및 끝 β 값
    -----------------------------------------
    과정:
     1) 시간 t를 0~1 사이로 정규화하고, 지수함수를 적용하여
        β(t) = beta_start * (beta_end/beta_start)^(t) 를 계산.
     2) α 값은 1 - β로 계산하고, 누적곱(alphas_cumprod)을 구함.
        (초기 α₀ = 1.0을 prepend하여 길이를 timesteps+2로 맞춤)
     3) 각 단계별 α 비율: alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]
     4) 최종 βₜ = 1 - (αₜ / αₜ₋₁)를 반환
    """
    # timesteps+1개의 점 (0부터 1까지)
    t = np.linspace(0, 1, timesteps+1)
    # 지수 스케줄: beta(t) = beta_start * (beta_end/beta_start)^t
    betas_cont = beta_start * ((beta_end / beta_start) ** t)
    alphas = 1.0 - betas_cont
    alphas_cumprod = np.cumprod(alphas)
    # 누적곱 맨 앞에 1을 붙여 길이를 timesteps+1로 맞춤
    alphas_cumprod = np.concatenate([np.array([1.0]), alphas_cumprod])
    alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas_discrete = 1 - alphas_step
    return betas_discrete

def sigmoid_beta_schedule_discrete(timesteps: int, beta_min: float = 0.0001, beta_max: float = 0.02,
                                   tau: float = 0.5, k: float = 10.0):
    """
    Sigmoid Scheduler (Discrete Version)
    -------------------------------------
    - timesteps: 전체 타임스텝 수 (출력 β 배열의 길이는 timesteps+1)
    - beta_min, beta_max: 최소/최대 β 값
    - tau: 시프트 파라미터 (중간 기준)
    - k: 스케일링 파라미터 (sigmoid의 기울기)
    -------------------------------------
    과정:
     1) 시간 t를 0~1 사이로 정규화하고, sigmoid 함수를 적용하여
        S(t) = 1 / (1 + exp(-k * (t - tau))) 를 계산.
     2) β(t) = beta_min + (beta_max - beta_min) * S(t) 로 β 값을 결정.
     3) 이후 α, 누적곱, 단계별 비율을 통해 βₜ = 1 - (αₜ / αₜ₋₁)를 반환
    """
    # timesteps+1개의 점 (0부터 1까지)
    t = np.linspace(0, 1, timesteps+1)
    # sigmoid 함수 적용: S(t) = 1 / (1 + exp(-k*(t-tau)))
    sig = 1.0 / (1.0 + np.exp(-k * (t - tau)))
    betas_cont = beta_min + (beta_max - beta_min) * sig
    alphas = 1.0 - betas_cont
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod = np.concatenate([np.array([1.0]), alphas_cumprod])
    alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas_discrete = 1 - alphas_step
    return betas_discrete
    
def linear_beta_schedule_discrete(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    Linear Scheduler (Discrete Version)
    ------------------------------------
    - timesteps: 전체 타임스텝 수 (최종 출력 β 배열의 길이는 timesteps+1)
    - beta_start, beta_end: 선형 스케줄의 시작 및 끝 β 값
    ------------------------------------
    과정:
     1) timesteps+1 개의 선형 β 값을 생성
            betas = linspace(beta_start, beta_end, timesteps+1)
     2) α 값은 1 - β로 계산하고, 누적곱(alphas_cumprod)을 구함.
            (초기 값 1.0을 prepend하여 길이를 timesteps+2로 만듦)
     3) 각 단계별 α 비율: alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]  
           → 이 결과의 길이는 timesteps+1
     4) βₜ = 1 - (αₜ / αₜ₋₁)를 반환
    """
    betas = np.linspace(beta_start, beta_end, timesteps+1)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod = np.concatenate([np.array([1.0]), alphas_cumprod])
    alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas_discrete = 1 - alphas_step
    return betas_discrete

def ddim_beta_schedule_discrete(timesteps: int, beta_start: float = 0.00085, beta_end: float = 0.012):
    """
    DDIM Scheduler (Discrete Version)
    -----------------------------------
    DDIM은 결정적 샘플링(η=0)을 위한 스케줄로, 보통 DDPM에서 사용하는 β 스케줄을 약간 다르게 
    설정합니다. 이 함수는 DDIM에서 주로 사용되는 beta_start, beta_end 값을 기반으로
    선형 β 스케줄을 만든 뒤, Linear Scheduler와 동일한 방식으로 각 단계별 
    βₜ = 1 - (αₜ / αₜ₋₁)를 계산합니다.
    최종 출력은 길이가 timesteps+1인 β 배열입니다.
    """
    betas = np.linspace(beta_start, beta_end, timesteps+1)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod = np.concatenate([np.array([1.0]), alphas_cumprod])
    alphas_step = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas_ddim = 1 - alphas_step
    return betas_ddim

def clip_noise_schedule(alphas2, clip_value=0.001):
    """(기존 함수 그대로) α² 시퀀스를 받아 단계별 비율을 clip 한 뒤 누적곱으로 복원"""
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2

def scaled_cosine_beta_schedule_discrete(
    timesteps: int,
    s: float = 0.008,
    initial_scale: float = 0.01,
    final_scale: float = 2.0,
    raise_to_power: float = 2.0
):
    """
    - timesteps 길이에 맞춘 원래의 cosine 베타 스케줄 (betas_full, 길이=timesteps+1) 을 구합니다.
    - 그중 인덱스 0..(timesteps-1) 구간에는 initial_scale → final_scale 스케일을 선형(혹은 비선형) 적용.
    - 인덱스 timesteps(마지막)에는 final_scale=1.0을 곱해 원본을 그대로 둡니다.
    → 결과 배열 길이 = timesteps+1
    """

    # 1) 원본 코사인 베타 전체 계산 (길이 = timesteps+1)
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])    # 길이 = steps-1 = timesteps+1
    betas_full = 1.0 - alphas                              # 길이 = timesteps+1
    betas_full = np.clip(betas_full, 0.0, 0.999)

    # 2) 스케일 계수 배열 만들기
    #    인덱스 0..(timesteps-1): linspace(initial_scale → final_scale, timesteps) ** raise_to_power
    #    인덱스 timesteps: final_scale (=1.0)
    scales = np.linspace(initial_scale, final_scale, timesteps) ** raise_to_power
    scales_full = np.concatenate([scales, [final_scale]])  # 길이 = timesteps+1

    # 3) betas_full 에 스케일을 곱한 뒤 클리핑
    betas_scaled_full = betas_full * scales_full
    betas_scaled_full = np.clip(betas_scaled_full, 0.0, 0.999)

    return betas_scaled_full
def scaled_poly_beta_schedule_discrete(timesteps: int,
                                      s: float = 1e-4,
                                      power: float = 3.,
                                      initial_scale: float = 0.01,
                                      final_scale: float = 2.0,
                                      raise_to_power: float = 2.0,
                                      clip_value: float = 0.001):

    steps = timesteps + 2 
    x = np.linspace(0, steps, steps)
    alphas2_cumprod = (1 - np.power(x / steps, power)) ** 2

    alphas2_cumprod = clip_noise_schedule(alphas2_cumprod, clip_value=clip_value)

    precision = 1 - 2 * s
    alphas2_cumprod = precision * alphas2_cumprod + s

    alphas_step = alphas2_cumprod[1:] / alphas2_cumprod[:-1]
    betas_full = 1.0 - alphas_step
    betas_full = np.clip(betas_full, 0.0, 0.999)

    scales = np.linspace(initial_scale, final_scale, timesteps) ** raise_to_power
    scales_full = np.concatenate([scales, [final_scale]])

    betas_scaled_full = betas_full * scales_full
    betas_scaled_full = np.clip(betas_scaled_full, 0.0, 0.999)

    return betas_scaled_full
def polynomial_beta_schedule_discrete(timesteps: int,
                                      s: float = 1e-4,
                                      power: float = 3.,
                                      clip_value: float = 0.001):
    """
    - polynomial(1 - x^power)^2 형태로 α²_cumprod를 만든 뒤
    - clip_noise_schedule 로 스텝 비율을 안정화하고
    - precision 보정 후
    - β_t = 1 - (α_t / α_{t-1}) 로 변환
    최종 β 배열 길이 = timesteps + 1   (cosine_beta_schedule_discrete 와 동일)
    """
    # 1) α²_cumprod 생성
    steps = timesteps + 2                     # cosine 쪽과 맞추기 위해 +2
    x = np.linspace(0, steps, steps)
    alphas2_cumprod = (1 - np.power(x / steps, power)) ** 2

    # 2) 단계별 비율 clip → 누적곱 재계산
    alphas2_cumprod = clip_noise_schedule(alphas2_cumprod, clip_value=clip_value)

    # 3) precision 보정 (원본 polynomial_schedule 과 동일)
    precision = 1 - 2 * s
    alphas2_cumprod = precision * alphas2_cumprod + s #todo s의 역할은?

    # 4) β 계산 :  β_t = 1 - (α_t / α_{t-1})
    #    (α² 사용 중이므로 √가 필요 없고, 비율을 그대로 쓰면 된다)
    alphas_step = alphas2_cumprod[1:] / alphas2_cumprod[:-1]
    betas = 1 - alphas_step
    return betas.squeeze()
    
def cosine_beta_schedule_discrete(timesteps, s: float = 0.008):#=0): s 가 줄어들면 알파가 커지고 베타가 작아짐
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def clipcosine_beta_schedule_discrete(timesteps, s: float = 0.008, clip_value : float = 0.001):#=0): s 가 줄어들면 알파가 커지고 베타가 작아짐
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = clip_noise_schedule(alphas_cumprod, clip_value=clip_value)
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=30, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()



def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_discrete_features(probX, probE, node_mask, step=None, add_nose=True):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1)      # (bs * n, 1)
    X_t = X_t.reshape(bs, n)        # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)           # (bs * n * n, de_out)
    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)    # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    X_t = X_t.float()
    Qt_T = Qt.transpose(-1, -2).float()                                       # bs, N, dt
    assert Qt.dim() == 3
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)                    # bs, N, 1, d_t-1
    right_term = Qsb.unsqueeze(1) 
    numerator = left_term * right_term                        # bs, N, d0, d_t-1
    
    denominator = Qtb @ X_t.transpose(-1, -2)                 # bs, d0, N
    denominator = denominator.transpose(-1, -2)               # bs, N, d0
    denominator = denominator.unsqueeze(-1)                   # bs, N, d0, 1

    denominator[denominator == 0] = 1.
    return numerator / denominator


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X.clamp_min(1e-18)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E.clamp_min(1e-18)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.ones(true_X.size(-1), dtype=true_X.dtype, device=true_X.device)
    row_E = torch.zeros(true_E.size(-1), dtype=true_E.dtype, device=true_E.device).clamp_min(1e-18)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    return true_X, true_E, pred_X, pred_E

def posterior_distributions(X, X_t, Qt, Qsb, Qtb, X_dim):
    bs, n, d = X.shape
    X = X.float()
    Qt_X_T = torch.transpose(Qt.X, -2, -1).float()                  # (bs, d, d)
    left_term = X_t @ Qt_X_T                                        # (bs, N, d)
    right_term = X @ Qsb.X                                          # (bs, N, d)
    
    numerator = left_term * right_term                              # (bs, N, d)
    denominator = X @ Qtb.X                                         # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denominator = denominator * X_t 
    
    num_X = numerator[:, :, :X_dim]
    num_E = numerator[:, :, X_dim:].reshape(bs, n*n, -1)

    deno_X = denominator[:, :, :X_dim]
    deno_E = denominator[:, :, X_dim:].reshape(bs, n*n, -1)

    # denominator = (denominator * X_t).sum(dim=-1)                   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    denominator = denominator.unsqueeze(-1)                         # (bs, N, 1)

    deno_X = deno_X.sum(dim=-1).unsqueeze(-1)
    deno_E = deno_E.sum(dim=-1).unsqueeze(-1)

    deno_X[deno_X == 0.] = 1
    deno_E[deno_E == 0.] = 1
    prob_X = num_X / deno_X
    prob_E = num_E / deno_E
    
    prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True)
    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True)
    return PlaceHolder(X=prob_X, E=prob_E, y=None)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).float()

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).float()

    U_X = U_X.to(node_mask.device)
    U_E = U_E.to(node_mask.device)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, y=None).mask(node_mask)

def index_QE(X, q_e, n_bond=5):
    bs, n, n_atom = X.shape
    node_indices = X.argmax(-1)  # (bs, n)

    exp_ind1 = node_indices[ :, :, None, None, None].expand(bs, n, n_atom, n_bond, n_bond)
    exp_ind2 = node_indices[ :, :, None, None, None].expand(bs, n, n, n_bond, n_bond)
    
    q_e = torch.gather(q_e, 1, exp_ind1)
    q_e = torch.gather(q_e, 2, exp_ind2) # (bs, n, n, n_bond, n_bond)


    node_mask = X.sum(-1) != 0
    no_edge = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
    q_e[no_edge] = torch.tensor([1, 0, 0, 0, 0]).type_as(q_e)

    return q_e
