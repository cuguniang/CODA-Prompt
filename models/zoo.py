import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
from .moco import vit_base as moco_base
from .dinov2.dinov2 import vit_base as dinov2_base
import numpy as np
import copy

#产生和挑选prompt 且 计算prompt带来的loss

class MyPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.flag = "my"
        self.task_count = 0
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # create and init prompt for each task
        # p = create_prompt_with_init(self.prompt_length_per_task * self.n_tasks, emb_d, mean=0.02391728200018406, std=0.478795012830553) # vit
        # p = create_prompt_with_init(self.prompt_length_per_task * self.n_tasks, emb_d, mean=-0.004702982492744923, std=0.02751666121184826) # moco
        p = create_prompt_with_init(self.prompt_length_per_task * self.n_tasks, emb_d)
        setattr(self, f'prompts', p)
        
        # final layer prompt weight init
        # w = create_prompt_with_init(self.n_tasks + 1, emb_d)
        # setattr(self, f'output_feature_weight', w)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True
        
        self.prompt_location = "input" #prompt_param[0] # input/attention

        # self.DEEP = bool(prompt_param[1])
        # print("prompt_param[1] = ",prompt_param[1])
        # print("bool(prompt_param[1]) = ",bool(prompt_param[1]))
        # self.SHARED = bool(prompt_param[2])    
        # print("prompt_param[2] = ",prompt_param[2])
        # print("bool(prompt_param[2]) = ",bool(prompt_param[2]))
        # prompt length per task
        self.prompt_length_per_task = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        B, _, _ = x_block.shape
        # select prompts and fixed previous prompt
        task_p_count = (self.task_count + 1) * self.prompt_length_per_task
        P = getattr(self, 'prompts')[:task_p_count,:]
        # 固定住之前任务的prompt；不固定则注释下面这行
        # P = torch.cat((P[:task_p_count-self.prompt_length_per_task].detach().clone(),P[-self.prompt_length_per_task:]), dim=0)         
    
        P = P.expand((B, -1, self.emb_d))
        
        # return
        return P, 0, x_block


# Our method!
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.flag = "coda"
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = create_prompt_with_init(self.e_pool_size, e_l, emb_d)
            k = create_prompt_with_init(self.e_pool_size, self.key_d)
            a = create_prompt_with_init(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.flag = "dual"
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = create_prompt_with_init(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = create_prompt_with_init(self.e_pool_size, self.e_p_length, emb_d)
            k = create_prompt_with_init(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        self.flag = "l2p"

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

# note - ortho init has not been found to help l2p/dual prompt
def create_prompt_with_init(a, b, c=None, ortho=False, mean=None, std=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    elif mean and std:
        nn.init.normal_(p, mean=mean, std=std)
    else:
        nn.init.uniform_(p)
    return p

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(ViTZoo, self).__init__()
        self.num_classes = num_classes
        # get last layer
        # self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
    
        self.tasks = tasks

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)
        else:
            pass
        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'my':
            self.prompt = MyPrompt(768, prompt_param[0], prompt_param[1])
            # task identifier weighted mapping
            # self.weight_mapping = nn.Linear(768, prompt_param[0]+1) # 做output的加权
            # self.feature_consist_mapping = nn.Linear(768, 768) # mean feature map 到 prompt
            # self.task_identifier = nn.Linear(768, prompt_param[0])  # 做task identifier

        else:
            self.prompt = None
        
        if self.prompt_flag == "my":
            tuned_params = ["prompt.prompts", # prompt
            "last.weight","last.bias", # 分类头
            # "weight_mapping.weight","weight_mapping.bias",# avg pooling feature mapping
            # "task_identifier.weight","task_identifier.bias",
            # "feature_consist_mapping.weight","feature_consist_mapping.bias"
            ] 
                    
            for name, param in self.named_parameters():
                if name in tuned_params:
                    param.requires_grad = True
                    print("tuneddddd parammmmm:",name)
                else:
                    param.requires_grad = False
                    # print("fixedddd parammmmm:",name)
        
        mean_value = torch.mean(self.feat.cls_token)
        std_value = torch.std(self.feat.cls_token)
        print("init cls token Mean:", mean_value.item())
        print("init cls token std:", std_value.item())

        # mean_value = torch.mean(self.prompt.prompts)
        # std_value = torch.std(self.prompt.prompts)
        # print("init prompts token Mean:", mean_value.item())
        # print("init prompts token std:", std_value.item())

        # prompt_len, _ = self.prompt.prompts.shape
        # for i in range(prompt_len):
        #     mean_value = torch.mean(self.prompt.prompts[i,:])
        #     std_value = torch.std(self.prompt.prompts[i,:])
            # print(f"init prompt token {i} Mean:", mean_value.item())
            # print(f"init prompt token {i} std:", std_value.item())

    def dist_loss(self, x1, x2, loss_type="L1"):

        if loss_type == "L1":
            return F.l1_loss(x1, x2)
        elif loss_type == "L2":
            return F.mse_loss(x1, x2)
        elif loss_type == "cos":
            # 将张量归一化为单位向量
            x1_normalized = F.normalize(x1, p=2, dim=-1)
            x2_normalized = F.normalize(x2, p=2, dim=-1)
            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(x1_normalized, x2_normalized, dim=-1)
            # 计算余弦相似度损失，可以使用 1 减去余弦相似度
            cls_cosine_similarity_loss = 1 - cls_cosine_similarity
            # 计算每个批次的平均损失
            batch_mean_loss = torch.mean(cls_cosine_similarity_loss)
            return batch_mean_loss
    
    def prompt_similiarity(self, p1, p2):
        # cos 
        # 0304 减去一个均值
        # p1 = p1 - p1.mean()
        # p2 = p2 - p2.mean()
        # 将张量归一化为单位向量
        p1_normalized = F.normalize(p1, p=2, dim=-1)
        p2_normalized = F.normalize(p2, p=2, dim=-1)
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(p1_normalized, p2_normalized, dim=-1)
        
        return cosine_similarity

    def get_attn_score_within_heads(self, attn_matrix, dim, method="mean"):
        if method == "mean":
            return attn_matrix.mean(dim=dim)

        elif method == "max":
            return attn_matrix.max(dim=dim)[0]

    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):
        # 只有prompt loss和head有梯度
        if self.prompt is not None:
            attn_loss_flag = False

            # weighted output
            weighted_output_flag = False
            # 蒸馏相关
            attn_within_prompt_pd_flag = True
            distill_flag = False
            cls_dist_flag = True
            mean_feature_dist_flag = True
            original_dist_flag = False
            prompts_progressive_dist_flag = True
            
                # select 相关
            SELECT_FLAG = False
            attn_within_heads = "max" # head间用mean/max
            FIXED_SELECT_FLAG = False  # false用limit（per sample）
            # 0306 用prompt i 的 attn 去select image tokens
            SELECT_TOKEN_NUMBER_1 = 196
            SELECT_TOKEN_NUMBER_2 = 196
            LIMIT_CLS = 0.01 #  attn score limit to decide number
            LIMIT_PROMPT = 0.01 #  attn score limit to decide number
            
            #0312 用attn i 做pd
            prompt_attn_pd_loss=0
            if self.prompt_flag == 'my':
              
                out, prompt_loss, attn = self.feat(x, prompt=self.prompt, train=train, task_id=self.task_id)
                B, _, emb_d = out.shape
                if weighted_output_flag:
                    # 0312 用attn score做加权
                    out_image_tokens = out[:,2+self.task_id:,:]
                    selected_tokens_weighted_mean = torch.zeros_like(out[:,:self.task_id+1,:], device=out.device)
                    for i in range(0, self.task_id+1):
                        attn2image_cls = self.get_attn_score_within_heads(attn[:,:,0,-196:], dim=1, method=attn_within_heads) # B, 196
                        # prompt i 对 196 token的注意力
                        attn2image_per_prompt = self.get_attn_score_within_heads(attn[:,:,i+1,-196:], dim=1, method=attn_within_heads)
                        # attn2prompt_per_token = self.get_attn_score_within_heads(attn[:,:,-196:,i+1], dim=1, method=attn_within_heads)
                        
                        token_weights = torch.softmax(attn2image_per_prompt, dim=-1) # b,196
                        selected_tokens_weighted_mean[:,i] = (out_image_tokens * token_weights.unsqueeze(dim=-1)).sum(dim=1)
                    
                    out = selected_tokens_weighted_mean
                else:
                    # out = out[:,0,:]
                    out = out[:,self.task_id+2:,:].mean(dim=1) # mean

                if SELECT_FLAG:            
                    #初始化结果
                    selected_tokens_mean = torch.zeros_like(out[:,:self.task_id+1,:], device=out.device)
                    out_image_tokens = out[:,2+self.task_id:,:]

                    # 处理每个task对应的prompt
                    for i in range(0, self.task_id+1):
                        if FIXED_SELECT_FLAG:
                            # 先过滤prompt 再 cls
                            attn2image_cls = self.get_attn_score_within_heads(attn[:,:,0,-196:], dim=1, method=attn_within_heads) # B, 196
                            attn2image_per_prompt = self.get_attn_score_within_heads(attn[:,:,i+1,-196:], dim=1, method=attn_within_heads)
                            # 用prompt的attn 过滤
                            tmp, sorted_indices = torch.sort(attn2image_per_prompt, dim=-1, descending=True) # b, 196
                            top_k_indices = sorted_indices[:,:SELECT_TOKEN_NUMBER_1] # b, num
                            selected_tokens_1 = torch.gather(out_image_tokens, dim=-2, index=top_k_indices.unsqueeze(dim=-1).expand(B, SELECT_TOKEN_NUMBER_1, emb_d)) # b,num.,768
                            attn2image_cls = torch.gather(attn2image_cls, dim=-1, index=top_k_indices)
                            # 用cls的attn 过滤
                            tmp, sorted_indices = torch.sort(attn2image_cls, dim=-1, descending=True)
                            top_k_indices = sorted_indices[:,:SELECT_TOKEN_NUMBER_2].unsqueeze(dim=-1).expand(B, SELECT_TOKEN_NUMBER_2, emb_d)
                            top_k_image_tokens = torch.gather(selected_tokens_1, dim=-2, index=top_k_indices)
                            selected_tokens_mean[:,i] = top_k_image_tokens.mean(dim=1)
            
                        else:# 根据attn的阈值动态决定数量
                            attn2image_cls = self.get_attn_score_within_heads(attn[:,:,0,-196:], dim=1, method=attn_within_heads) # B, 196
                            attn2image_per_prompt = self.get_attn_score_within_heads(attn[:,:,i+1,-196:], dim=1, method=attn_within_heads)
                            #交集
                            selected_mask = attn2image_cls.gt(LIMIT_CLS) * attn2image_per_prompt.gt(LIMIT_PROMPT)
                            # print("limit tokens:",selected_mask.sum())

                            for bi in range(B):
                                selected_num = selected_mask[bi].sum()
                                # a  = out_image_tokens[bi].mean(dim=0)
                                b = (out_image_tokens[bi] * selected_mask[bi].unsqueeze(dim=1)).sum(dim=0) / selected_num
                                # selected_tokens_mean[bi,i] = (out_image_tokens[bi] * selected_mask[bi].unsqueeze(dim=1)).sum() / selected_num
                                selected_tokens_mean[bi,i] = b

                    out = selected_tokens_mean
                
                # ===attn pd loss =====
                if attn_within_prompt_pd_flag and self.task_id>0:
                # 之前每个prompt及其对应的prompted_feature
                    prompt_of_now_task = self.prompt.prompts[self.task_id]
                    prompt_now_attn = self.get_attn_score_within_heads(attn[:,:,self.task_id+1,-196:], dim=1, method=attn_within_heads) # B, 196
                    for i in range(0, self.task_id):
                        prompt_of_task_i = self.prompt.prompts[i]
                        prompt_similiarity = self.prompt_similiarity(prompt_of_task_i, prompt_of_now_task)
                        if prompt_similiarity <= 0:
                            continue
                        # print(f"prompt_similiarity {i} & {self.task_id}: {prompt_similiarity}" )
                        prompt_i_attn = self.get_attn_score_within_heads(attn[:,:,i+1,-196:], dim=1, method=attn_within_heads) # B, 196
                        
                        prompt_attn_pd_loss += prompt_similiarity * self.dist_loss(prompt_i_attn, prompt_now_attn, "L1")
      
                # ===0302 attn===
                if attn_loss_flag:
                    attn_per_prompt = attn[:,:,1,:].max(dim=-1)[0].mean(dim=1,keepdim=True) # bx1
                    # print("attn_per_prompt 1 :",attn_per_prompt[0].item())

                    for i in range(2, self.task_id+2):
                        attn_per_prompt = torch.cat((attn_per_prompt, torch.max(attn[:,:,i,:], dim=-1)[0].mean(dim=1,keepdim=True)), dim=1) # bxi
                        # print(f"attn_per_prompt {i} :",attn_per_prompt[0,i-1].item())

                    if train:
                        targets_task_ids = torch.full((B,), self.task_id, device=attn_per_prompt.device)
                        attn_weights = torch.softmax(attn_per_prompt, dim=1) 
                        # 定义 CrossEntropyLoss 损失函数
                        criterion = nn.CrossEntropyLoss()
                        # 计算损失
                        attn_loss = criterion(attn_weights, targets_task_ids)
                    # print("attn argmax:", attn_per_prompt.argmax(dim=1))
                # ===0302 attn end===

                if distill_flag:
                    q, _, _ = self.feat(x)
                    original_cls = q[:,0,:]
                    original_avg_pooled_feature = q[:,1:,:].mean(dim=1)
                    prompted_pooled_feature = out[:,2+self.task_id:,:].mean(dim=1) # [17, 768]
                    prompted_cls = out[:,0,:]

                    if cls_dist_flag:
                        cls_distill_loss = 0
                    if mean_feature_dist_flag:
                        mean_feature_distll_loss = 0
                    
                    # ===2d=== 先做预训练模型和当前的prompts 的蒸馏
                    if original_dist_flag:
                        cls_distill_loss += self.dist_loss(prompted_cls, original_cls, "L1")
                    if mean_feature_dist_flag:
                        mean_feature_distll_loss += self.dist_loss(prompted_pooled_feature, original_avg_pooled_feature, "L1")
                    
                    # ===0303 panda=== 融入prompt相似度的逐步蒸馏
                    if prompts_progressive_dist_flag and self.task_id>0:
                    # 之前每个prompt及其对应的prompted_feature
                        prompt_of_now_task = self.prompt.prompts[self.task_id]
                        for i in range(0, self.task_id):
                            prompt_of_task_i = self.prompt.prompts[i]
                            prompt_similiarity = self.prompt_similiarity(prompt_of_task_i, prompt_of_now_task)
                            if prompt_similiarity <= 0:
                                continue
                            # print(f"prompt_similiarity {i} & {self.task_id}: {prompt_similiarity}" )
                            prompted_out_until_i, _, _ = self.feat(x, prompt=self.prompt, train=train, task_id=i)
                            
                            cls_i = prompted_out_until_i[:,0,:]
                            mean_feature_i = prompted_out_until_i[:,i+2:].mean(dim=1)
                            
                            if cls_dist_flag:
                                cls_distill_loss += prompt_similiarity * self.dist_loss(prompted_cls, cls_i, "L1")
                            if mean_feature_dist_flag:
                                mean_feature_distll_loss += prompt_similiarity * self.dist_loss(prompted_pooled_feature, mean_feature_i, "L1")

                    total_distill_loss = 0
                    if original_dist_flag or self.task_id>0:
                        if cls_dist_flag:
                            total_distill_loss += cls_distill_loss
                        if mean_feature_dist_flag:
                            total_distill_loss += mean_feature_distll_loss
               
                # ===mapping weight===

                # mapping_feature = self.feature_consist_mapping(avg_pooled_feature)

                # if train:
                    # mse_feature_mapping_loss = F.mse_loss(out[:,1+self.task_id,:], mapping_feature).mean(dim=0)
                # task_id_preds = self.task_identifier(avg_pooled_feature)[:,:self.task_id+1]
                # out_cls_weights = self.weight_mapping(avg_pooled_feature)[:,:self.task_id+2] #[17, 11]
                # cls_prompt_tokens = out[:,:2+self.task_id,:]
                # out = torch.sum(cls_prompt_tokens * out_cls_weights.unsqueeze(-1), dim=1) #[17, 768]
                
                # ===print output===
                # mean_value = torch.mean(out[:,0,:],dim=1).mean()
                # std_value = torch.std(out[:,0,:],dim=1).mean()
                # print("output cls token Mean:", mean_value.item())
                # print("output cls token std:", std_value.item())

                # mean_value = torch.mean(out[:,1:self.task_id+2,:], dim=2).mean()
                # std_value = torch.std(out[:,1:self.task_id+2,:], dim=2).mean()
                # print("output prompts token Mean:", mean_value.item())
                # print("output prompts token std:", std_value.item())

                # prompt_len, _ = self.prompt.prompts.shape
                # for i in range(prompt_len):
                #     mean_value = torch.mean(out[:,1+i,:],dim=1)
                #     std_value = torch.std(out[:,1+i,:],dim=1)
                #     print(f"output prompts token {i} Mean:", mean_value)
                #     print(f"output prompts token {i} std:", std_value)
                
                # 采用不同的输出去做最后的分类
                # out = torch.cat((out[:,0,:].unsqueeze(dim=1),out[:,1+self.task_id,:].unsqueeze(dim=1)), dim=1)
                # out = out.mean(dim=1) # (0+i)/2
                # print("out mean,",out.shape)

                # out = out[:,1+self.task_id,:] # [i]
                # out = out[:,0,:] #[0]
                # out = out[:,1:self.task_id+2,:] # prompts
                # out = out[:,self.task_id+2:,:].mean(dim=1) # mean
                # out = out[:,1:self.task_id+2,:].mean(dim=1) # pa
                # out = out[:,:self.task_id+2,:].mean(dim=1) # cpa
                # out_image_tokens = out[:,2+self.task_id:,:]
                # out = torch.cat((out[:,0:1,:], out_image_tokens),dim=1).mean(dim=1, keepdim=True) # cls + mean .mean
                # 0301 prompt + mean .mean
                # # '''
                # prompted_features = torch.cat((out[:,0:2,:], out_image_tokens),dim=1).mean(dim=1, keepdim=True)
                # # print("prompted_features",prompted_features.shape)
                # for i in range(2, self.task_id+2):
            
                #     # temp_i_features = torch.cat((out[:,i:i+1,:], out_image_tokens), dim=1).mean(dim=1,keepdim=True) # (pi+mean).mean
                #     temp_i_features = torch.cat((out[:,0:1,:], out[:,i:i+1,:], out_image_tokens), dim=1).mean(dim=1,keepdim=True) # (cls+pi+mean).mean
                #     prompted_features = torch.cat((prompted_features, temp_i_features), dim=1)
                
                # out = prompted_features
                # ====0301 end====
                # '''
                # else:
                #     q = nn.functional.normalize(mapping_feature).unsqueeze(dim=1)
                #     k = nn.functional.normalize(out[:,1:self.task_id+2,:])
                #     cos_sim = torch.einsum('btj,kpj->btp', q, k) # ([b, 1, 10])
                #     # q.shape torch.Size([32, 1, 768])
                #     # k.shape torch.Size([32, 1, 768])
                #     # cos_sim.shape torch.Size([32, 1, 1])
                #     print("q.shape",q.shape)
                #     print("k.shape",k.shape)
                #     print("cos_sim.shape",cos_sim.shape)
                #     predicted_prompt_ids = torch.argmax(cos_sim.squeeze(dim=1), dim=1)
                #     predicted_prompt_ids += 1
                #     out = out[torch.arange(B), predicted_prompt_ids,:]
                #     print("predicted_prompt_ids",predicted_prompt_ids)
               
            else:
                with torch.no_grad():
                    q, _, _ = self.feat(x)
                    q = q[:,0,:]
                    out, prompt_loss, _ = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
                    out = out[:,0,:]
        else:
            out, _, _ = self.feat(x)
            out = out[:,0,:]
        if out.size(1) == 1:
            out = out.view(out.size(0), -1) # 如果out是cls
        # =====print last=====
        # mean_value = torch.mean(self.last.weight)
        # std_value = torch.std(self.last.weight)
        # print("last classification head weight Mean:", mean_value.item())
        # print("last classification head weight std:", std_value.item())    

        # class_num, dim = self.last.weight.shape
        # for i in range(0, class_num, 10):
        #     mean_value = torch.mean(self.last.weight[i:i+10])
        #     # std_value = torch.std(self.last.weight[i:i+10])
        #     print(f"last classification head weight of task {i//10+1} Mean: ", mean_value.item())
            # print(f"last classification head weight of task {i//10+1} std: ", std_value.item())
        # =====================
        if not pen:
            out = self.last(out) # B x 100  
            # '''
            print("len(out.size)",len(out.size))
            if len(out.size) > 2:
                out_logits = out[:,0,:len(self.tasks[0])]
                num_classes_cnt = len(self.tasks[0])
                # attn_weights = torch.softmax(attn_per_prompt, dim=1) # attn
                # out_logits = out[:,0,:10] * attn_weights[:,0].unsqueeze(dim=1) # attn
                
                for i in range(1, self.task_id+1): 
                    # out_logits = torch.cat((out_logits, out[:,i,i*10:i*10+10]),dim=1)
                    out_logits = torch.cat((out_logits, out[:,i,num_classes_cnt:num_classes_cnt+len(self.tasks[i])]),dim=1)
                    num_classes_cnt += len(self.tasks[i])
                    # out_logits = torch.cat((out_logits, out[:,i,i*10:i*10+10]*attn_weights[:,i].unsqueeze(dim=1)),dim=1) # attn

                out = out_logits
            else:
                pass
                # attn_weights = torch.softmax(attn_per_prompt, dim=1) # attn
                # out_logits = out[:,0,:10] * attn_weights[:,0].unsqueeze(dim=1) # attn
                # for i in range(1, self.task_id+1): 
                #     # out_logits = torch.cat((out_logits, out[:,i,i*10:i*10+10]),dim=1)
                #     out_logits = torch.cat((out_logits, out[:,i,i*10:i*10+10]*attn_weights[:,i].unsqueeze(dim=1)), dim=1) # attn
           
                # # === logits sum
                # logits_sum = out[:,0,:10].sum(dim=1, keepdim=True)
                # # b 10 100
                # for i in range(1, self.task_id+1):
                    # print("sum",i,out[:,i,i*10:i*10+10].sum(dim=1,keepdim=True))
                    # logits_sum = torch.cat((logits_sum, out[:,i,i*10:i*10+10].sum(dim=1,keepdim=True)), dim=1)
                    # print(f"{i+1} mean ",out[:,i,i*10:i*10+10].mean(dim=1))
                    # print("out_logits",out_logits.shape)

                    # out_logits[:,i*10:i*10+10] = out[:,i,i*10:i*10+10]
                
                # print("logits_sum", logits_sum.shape) # 32 * 10

                # logits_sum_weight = torch.softmax(logits_sum, dim=1)
                # print("logits_sum_weight",logits_sum_weight) # 32 * 1
                # print("task id pred:",logits_sum_weight.argmax(dim=1))
                # 0229 test-2 多一步softmax
                # out = torch.softmax(out, dim=-1)
                # out_logits = out[:,0,:10] * logits_sum_weight[:,0].unsqueeze(dim=1)
                # print("out_logits init",out_logits.shape)
                # for i in range(1, self.task_id+1):
                #     weighted_logits = out[:,i,i*10:i*10+10] * (logits_sum_weight[:,i].unsqueeze(dim=1))
                #     print(" out[:,i,i*10:i*10+10]",out[:,i,i*10:i*10+10].shape)
                #     print("(logits_sum_weight[:,i].unsqueeze(dim=1))",(logits_sum_weight[:,i].unsqueeze(dim=1)).shape)
                #     print("weighted_logits",weighted_logits.shape)

                #     out_logits = torch.cat((out_logits, weighted_logits),dim=1)
                #=== end logits sum ==='''
            # # 0227-prompt分别给不同的weight
            # for i in range(self.task_id+1):
            #     out_prompt = out[:,i+1,:].unsqueeze(dim=1) # [32, 1, 768]
            #     task_classification_weight = self.last.weight[10*i:10*i+10] # 10*768
            #     task_classification_bias = self.last.bias[10*i:10*i+10] # 10
            #     logit_i = out_prompt @ task_classification_weight.transpose(-1,-2)  + task_classification_bias
            #     # print("logit_i1",logit_i.shape)
            #     logit_i = logit_i.squeeze(dim=1) # 32 * 10(class_num per task)
            #     # print("logit_2i",logit_i.shape)
            #     out_logits[:,10*i:10*i+10] = logit_i

            # task_identifier
            # predicted_task_id = torch.argmax(task_id_preds, dim=1)
            # print("predicted_task_id,",predicted_task_id)
            # '''
        if self.prompt is not None and train:
            if distill_flag:
                prompt_loss += total_distill_loss
            if attn_loss_flag:
                prompt_loss += attn_loss
            if attn_within_prompt_pd_flag:
                prompt_loss += prompt_attn_pd_loss
            return out, prompt_loss
        else:
            return out

class MoCoZoo(ViTZoo):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(MoCoZoo, self).__init__(num_classes, pt, prompt_flag, prompt_param, tasks)
       
        if pt:
            zoo_model = moco_base()#VisionTransformerMoCo(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                     #   num_heads=12,
                                    #    drop_path_rate=0
                                   #     )
            ckpt = "/share/ckpt/cgn/vpt/model/mocov3_linear-vit-b-300ep.pth.tar"

            checkpoint = torch.load(ckpt, map_location="cpu")
            load_dict = checkpoint['state_dict']
            for k in list(load_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    load_dict[k[len("module."):]] = load_dict[k]
                # delete renamed or unused k
                del load_dict[k]

            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)

        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

        mean_value = torch.mean(self.feat.cls_token)
        std_value = torch.std(self.feat.cls_token)
        print("moco init cls token Mean:", mean_value.item())
        print("moco init cls token std:", std_value.item())


class Dinov2Zoo(ViTZoo):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, tasks=[]):
        super(Dinov2Zoo, self).__init__(num_classes, pt, prompt_flag, prompt_param, tasks)
       
        if pt:
            zoo_model = dinov2_base(patch_size=14)
            ckpt = "/share/ckpt/cgn/vpt/model/dinov2_vitb14_reg4_pretrain.pth"

            checkpoint = torch.load(ckpt, map_location="cpu")
            load_dict = checkpoint#['state_dict']
            for k in list(load_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('blocks'):
                    # remove prefix
                    load_dict["blocks.0."+k[len("blocks."):]] = load_dict[k]
                    del load_dict[k]

            del load_dict['pos_embed']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict, strict=False)

        else:
            pass
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

        mean_value = torch.mean(self.feat.cls_token)
        std_value = torch.std(self.feat.cls_token)
        print("dinov2 init cls token Mean:", mean_value.item())
        print("dinov2 init cls token std:", std_value.item())


def vit_pt_imnet(out_dim, tasks=[], block_division = None, prompt_flag = 'None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)
    

def moco_pt(out_dim, tasks=[], block_division = None, prompt_flag = 'None', prompt_param=None):
    return MoCoZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)


def dino_pt(out_dim, tasks=[], block_division = None, prompt_flag = 'None', prompt_param=None):
    return Dinov2Zoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, tasks=tasks)
    