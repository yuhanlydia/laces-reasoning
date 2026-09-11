"""Original LACES S0/S1/S2 interface for direct S2 post-training.

No replacement encoder, writer, recurrent refiner or auxiliary answer classifier.
All label likelihoods use the frozen RWKV vocabulary head after native state writes.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import torch
from .data import LETTERS


def _active_groups(model):
    s0={'mlp':['encoder'],'variational':['encoder_trunk','mu_head','logvar_head'],
        'identity':['latent_mu','latent_sigma']}.get(model.encoder_type)
    if s0 is None: raise ValueError('Unsupported original S0 encoder')
    if getattr(model,'s0_input_adapter',None) is not None: s0=s0+['s0_input_adapter']
    writer=getattr(model,'s1_writer_type','fixed')
    s1={'dynlowrank':['s1_trunk','s1_u_head','s1_v_head','state_scale'],
        'fixed':['alpha_heads','state_basis','state_scale']}.get(writer)
    if s1 is None: raise ValueError('Supported writers: pretrained fixed or dynlowrank')
    if writer=='fixed' and getattr(model,'alpha_type','linear')!='linear': s1+=['alpha_trunk']
    if getattr(model,'use_learnable_blend',False): s1+=['blend_gate_logit']
    return dict(S0=s0,S1=s1,S2=['trajectory_dit'])


def validate_checkpoint(model,checkpoint,expected_step,expected_writer):
    if expected_step is not None and checkpoint.get('step')!=expected_step:
        raise ValueError(f'checkpoint step {checkpoint.get("step")} != {expected_step}')
    if model.s1_writer_type!=expected_writer: raise ValueError('Wrong writer type; refusing a silent fixed-basis fallback')
    if getattr(model,'trajectory_s1_mode',None)!='independent':
        raise ValueError('This entry targets the independent S1 interface; do not substitute another writer')
    current=model.state_dict(); saved=checkpoint.get('trainable_state',{}); fingerprint=sha256(); coverage={}
    for group,prefixes in _active_groups(model).items():
        keys=[]
        for prefix in prefixes:
            matching=[k for k in current if k==prefix or k.startswith(prefix+'.')]
            if not matching: raise ValueError(f'Missing active module {prefix}')
            keys+=matching
        for k in keys:
            if k not in saved: raise ValueError(f'Missing checkpoint tensor: {k}')
            a=current[k].detach().cpu(); b=saved[k].detach().to(device='cpu',dtype=a.dtype)
            if a.shape!=b.shape or not torch.equal(a,b): raise ValueError(f'Loaded tensor mismatch: {k}')
            fingerprint.update(k.encode()); fingerprint.update(str((a.shape,a.dtype)).encode())
            fingerprint.update(memoryview(a.contiguous().view(torch.uint8).numpy()))
        coverage[group]=len(keys)
    return dict(step=checkpoint.get('step'),writer_type=model.s1_writer_type,
                active_fingerprint=fingerprint.hexdigest(),keys=coverage,
                latent_dim=model.latent_dim,horizon=model.trajectory_horizon,chunk_size=model.trajectory_chunk_size)


class NativeLACES:
    def __init__(self,model,tokenizer,checkpoint,*,expected_step=30000,expected_writer='dynlowrank',blend=.7):
        self.audit=validate_checkpoint(model,checkpoint,expected_step,expected_writer)
        if not 0<=blend<=1: raise ValueError('blend outside [0,1]')
        if str(getattr(model,'_gen_type','ddpm'))!='ddpm': raise ValueError('Only epsilon-prediction DDPM LACES checkpoints supported')
        self.model=model.eval().requires_grad_(False); self.tokenizer=tokenizer; self.blend=float(blend)
        self.horizon=int(model.trajectory_horizon); self.chunk=int(model.trajectory_chunk_size)
        self.s2=model.trajectory_dit
        self.parent_s2_state={k:v.detach().cpu().clone() for k,v in self.s2.state_dict().items()}
        self.device=next(model.parameters()).device
        self.calls=Counter(S0=0,S1=0,S2=0); self.hooks=[]
        s0=model.mu_head if model.encoder_type=='variational' else getattr(model,'encoder',None)
        s1=model.s1_trunk if model.s1_writer_type=='dynlowrank' else model.alpha_heads[0]
        for group,mod in [('S0',s0),('S1',s1),('S2',self.s2)]:
            if mod is not None:
                def record(_m,_a,_out,group=group): self.calls[group]+=1
                self.hooks.append(mod.register_forward_hook(record))
        self.model._prefix_suffix_trajectory_s2=True; self.model._training_stage=2

    def close(self):
        for hook in self.hooks: hook.remove()
        self.hooks=[]

    def enable_s2_training(self):
        self.model.requires_grad_(False)
        # Optimizer/master parameters stay FP32; original S0/S1/RWKV dtypes unchanged.
        self.s2.float().requires_grad_(True).eval()
        selected={id(p) for p in self.s2.parameters()}
        actual={id(p) for p in self.model.parameters() if p.requires_grad}
        if not selected or actual!=selected: raise RuntimeError('Only pretrained S2 may be trainable')

    def _ids(self,text):
        return self.tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to(self.device)

    @torch.no_grad()
    def encode_prompt(self,text: str,*,seed: int):
        ids=self._ids(text)
        if ids.shape[1]<2: raise ValueError('Prompt must contain at least two tokens')
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            out=self.model.rwkv_model(input_ids=ids,attention_mask=torch.ones_like(ids).bool(),
                                      output_hidden_states=True,use_cache=True,return_dict=True)
            pooled=self.model._pool_hidden(out.hidden_states[-1],torch.ones_like(ids))
            cond,_=self.model._encode_pooled(pooled)
            if self.model.encoder_type=='identity': self.calls['S0']+=1
        if not torch.isfinite(cond).all(): raise FloatingPointError('Nonfinite native S0 conditioning')
        return ids,cond.detach().float()

    @torch.no_grad()
    def encode_teacher(self,text: str,*,seed: int):
        """Optional teacher-trace distillation in native isolated-chunk S0 coordinates.

        No invented hidden CoT: text must come from a supplied, attributed training
        rationale. Padding chunks are masked out of the denoising loss.
        """
        if any(bool(getattr(self.model,k,False)) for k in ('_s1_xchunk_enabled','_s1_global_anchor_enabled')):
            raise ValueError('Teacher encoding must be adapted for this non-independent S0 chunk context')
        ids=self._ids(text); length=ids.shape[1]
        if length<1 or length>self.horizon*self.chunk:
            raise ValueError(f'Teacher length {length} exceeds native horizon {self.horizon*self.chunk}; not truncating')
        z=torch.zeros(1,self.horizon,self.model.latent_dim,device=self.device); mask=torch.zeros(1,self.horizon,device=self.device)
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for h,start in enumerate(range(0,length,self.chunk)):
                chunk=ids[:,start:start+self.chunk]
                out=self.model.rwkv_model(input_ids=chunk,attention_mask=torch.ones_like(chunk).bool(),
                        output_hidden_states=True,use_cache=True,return_dict=True)
                pooled=self.model._pool_hidden(out.hidden_states[-1],torch.ones_like(chunk))
                zh,_=self.model._encode_pooled(pooled); z[:,h]=zh.float(); mask[:,h]=1
        return z.detach(),mask

    def _stream(self,prefix,z,length,*,raw=False):
        if prefix.shape[0]!=1 or prefix.shape[1]<2 or length<1: raise ValueError('Invalid sequence for scoring')
        if z.shape!=(1,self.horizon,self.model.latent_dim): raise ValueError('Wrong pretrained trajectory shape')
        if length>self.horizon*self.chunk: raise ValueError('Answer exceeds native horizon; not silently truncating')
        out=self.model.rwkv_model(input_ids=prefix[:,:-1],attention_mask=torch.ones_like(prefix[:,:-1]).bool(),
                                  use_cache=True,return_dict=True)
        cache=out.past_key_values; pending=prefix[:,-1:]
        for i in range(length):
            if i%self.chunk==0 and not raw and self.blend!=0:
                # Native dtype/scale/semantics. A planned state is NOT an additive delta.
                if self.model.s1_writer_type=='dynlowrank': dtype=next(self.model.s1_trunk.parameters()).dtype
                else: dtype=next(self.model.alpha_heads.parameters()).dtype
                states=self.model.predict_states(z[:,i//self.chunk].to(device=self.device,dtype=dtype))
                cache=(self.model.blend_into_cache(cache,states,self.blend) if self.blend<1
                       else self.model.inject_into_cache(cache,states))
            out=self.model.rwkv_model(input_ids=pending,past_key_values=cache,use_cache=True,return_dict=True)
            cache=out.past_key_values
            token=yield out.logits[0,-1].float()
            pending=torch.tensor([[int(token)]],device=self.device)

    @torch.no_grad()
    def score_tokens(self,prefix,answer,z,*,raw=False):
        answer=answer.to(self.device)
        stream=self._stream(prefix,z,answer.shape[1],raw=raw); logits=next(stream); scores=[]
        try:
            for i,token in enumerate(answer[0]):
                scores.append(torch.log_softmax(logits,-1)[token])
                if i+1<answer.shape[1]: logits=stream.send(int(token))
        finally: stream.close()
        return torch.stack(scores).sum() # full joint continuation likelihood, not substring match

    @torch.no_grad()
    def score_options(self,prefix,z,n_options: int,*,raw=False,prompt_text=None):
        if not 2<=n_options<=10: raise ValueError('2..10 choices required')
        continuations=[self._ids(' '+x) for x in LETTERS[:n_options]]
        if prompt_text is not None:
            for letter,suffix in zip(LETTERS[:n_options],continuations):
                full=self._ids(prompt_text+' '+letter)
                if not torch.equal(full,torch.cat([prefix,suffix],1)):
                    raise ValueError('Tokenizer boundary merges prompt/answer; use a stable explicit delimiter')
        # Common prefix + single terminal token: one prefill and one shared stream for A..J.
        # Handles character tokenizers (space, letter) and typical one-token labels efficiently.
        lists=[x[0].tolist() for x in continuations]; common=[]
        for i in range(min(map(len,lists))):
            if len({x[i] for x in lists})!=1: break
            common.append(lists[0][i])
        if all(len(x)==len(common)+1 for x in lists):
            stream=self._stream(prefix,z,len(common)+1,raw=raw); logits=next(stream); total=0.
            try:
                for token in common:
                    total=total+torch.log_softmax(logits,-1)[token]; logits=stream.send(token)
                logp=torch.log_softmax(logits,-1)
                return total+logp[torch.tensor([x[-1] for x in lists],device=self.device)]
            finally: stream.close()
        return torch.stack([self.score_tokens(prefix,x,z,raw=raw) for x in continuations])

    @torch.no_grad()
    def generate(self,prefix,z,max_tokens,*,raw=False):
        stream=self._stream(prefix,z,max_tokens,raw=raw); logits=next(stream); ids=[]
        try:
            for i in range(max_tokens):
                token=int(logits.argmax())
                if token==getattr(self.tokenizer,'eos_token_id',None): break
                ids.append(token)
                if i+1<max_tokens: logits=stream.send(token)
        finally: stream.close()
        return ids,self.tokenizer.decode(ids,skip_special_tokens=False)


def load_native(ckpt_dir,device='cuda',*,rwkv_path=None,expected_step=30000,expected_writer='dynlowrank',blend=.7):
    """Local user checkpoint only. Do not reinitialize or replace S0/S1/S2."""
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.state_hijacking_dit import StateInjectionDiTRELAY
    path=Path(ckpt_dir)/'model.pt'
    if not path.is_file(): raise FileNotFoundError(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    config=OmegaConf.create(checkpoint['config']); backbone=rwkv_path or config.model.rwkv_local_path
    dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32
    rwkv=AutoModelForCausalLM.from_pretrained(backbone,trust_remote_code=True,torch_dtype=dtype,local_files_only=True).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(backbone,trust_remote_code=True,local_files_only=True)
    model=StateInjectionDiTRELAY(config=config.model,rwkv_model=rwkv,vocab_size=len(tokenizer),
        latent_dim=int(config.model.latent_dim),n_basis=int(config.model.get('n_basis',16)),
        dit_hidden=int(config.model.get('dit_hidden',256)),dit_depth=int(config.model.get('dit_depth',4)),
        dit_num_heads=int(config.model.get('dit_num_heads',4)),dit_num_tokens=int(config.model.get('dit_num_tokens',4)),
        encoder_type=str(config.model.get('encoder_type','mlp')),alpha_type=str(config.model.get('alpha_type','linear')),
        alpha_hidden=int(config.model.get('alpha_hidden',256)),latent_stats_path=config.model.get('latent_stats_path',None)).to(device)
    incompatible=model.load_state_dict(checkpoint['trainable_state'],strict=False)
    for p in model.parameters():
        if p.requires_grad and p.is_floating_point(): p.data=p.data.to(dtype)
    model._gen_type=str(config.training.get('gen_type','ddpm'))
    native=NativeLACES(model,tokenizer,checkpoint,expected_step=expected_step,expected_writer=expected_writer,blend=blend)
    native.audit.update(backbone_path=str(backbone),checkpoint_path=str(path.resolve()),
                        load_missing_keys=list(incompatible.missing_keys),load_unexpected_keys=list(incompatible.unexpected_keys))
    return native
