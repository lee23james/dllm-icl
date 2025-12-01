import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def forward_process(batch, answer_index, mask_id):
    b, l = batch.shape

    target_len = answer_index.sum().item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    # Map the mask to the actual answer positions in the sequence
    is_mask_full = torch.zeros(b, l, dtype=torch.bool, device=batch.device)
    for i in range(b):
        is_mask_full[i, answer_index] = is_mask[i]
    
    noisy_batch = torch.where(is_mask_full, mask_id, batch)

    # Return the masked batch and the mask ratio (only for answer positions)
    p_mask = torch.zeros(b, l, device=batch.device)
    p_mask[:, answer_index] = (x / target_len).unsqueeze(1).repeat(1, target_len)
    return noisy_batch, p_mask


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input = batch
    logits = model(input).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@ torch.no_grad()
def get_log_likelihood(model, prompt_left, answer, prompt_right=None, mc_num=128, batch_size=16, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt_left: A tensor of shape (l1). The prompt tokens before answer (ice examples).
        answer: A tensor of shape (l2). The answer tokens in the middle.
        prompt_right: A tensor of shape (l3). The prompt tokens after answer (ice examples). 
                      If None, only prompt_left is used (backward compatible).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    device = answer.device
    if prompt_right is None:
        # Backward compatible: if prompt_right is None, treat prompt_left as the full prompt
        seq = torch.concatenate([prompt_left, answer])[None, :]
        prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt_left)
        answer_start = len(prompt_left)
        answer_end = answer_start + len(answer)
    else:
        # New structure: prompt_left + answer + prompt_right
        seq = torch.concatenate([prompt_left, answer, prompt_right])[None, :]
        answer_start = len(prompt_left)
        answer_end = answer_start + len(answer)
        prompt_index = torch.ones(seq.shape[1], dtype=torch.bool, device=device)
        prompt_index[answer_start:answer_end] = False
    
    seq = seq.repeat((batch_size, 1)).to(device)
    answer_index = torch.zeros(seq.shape[1], dtype=torch.bool, device=device)
    answer_index[answer_start:answer_end] = True

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, answer_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum() / batch_size

        loss_.append(loss.item())

    return - sum(loss_) / len(loss_)


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)

    # Example 1: Original usage (backward compatible)
    prompt = 'Roof shingle removal: A man is sitting on a roof. He'
    answer = ' is using wrap to wrap a pair of skis.'
    prompt = torch.tensor(tokenizer(prompt)['input_ids']).to(device)
    answer = torch.tensor(tokenizer(answer)['input_ids']).to(device)
    print("Original format:", get_log_likelihood(model, prompt, answer, mc_num=128))
    
    # Example 2: New format with prompt_left + answer + prompt_right
    prompt_left = 'Example 1: Some text. Example 2: Another text. Question:'
    prompt_right = ' Choose the best answer.'
    answer_new = ' The answer is A.'
    prompt_left = torch.tensor(tokenizer(prompt_left)['input_ids']).to(device)
    prompt_right = torch.tensor(tokenizer(prompt_right)['input_ids']).to(device)
    answer_new = torch.tensor(tokenizer(answer_new)['input_ids']).to(device)
    print("New format:", get_log_likelihood(model, prompt_left, answer_new, prompt_right, mc_num=128))


if __name__ == '__main__':
    main()