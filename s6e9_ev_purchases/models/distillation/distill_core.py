"""Student network: a parallel ensemble of MLPs (one weight tensor per member) on the standardized feature bank plus
the nested LR margin column, added to that margin as an offset; trained on the true labels of the fit rows plus a
weighted soft-label loss on the query rows."""
import numpy as np


def dense(A, types):
    columns = []
    for j, t in enumerate(types):
        if t == 'c':
            for value in range(1, 4):
                columns.append((A[:, j] == value).astype('float32'))
        else:
            columns.append(A[:, j])
    return np.column_stack(columns).astype('float32')


def make_model(dim, cfg, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Ensemble(nn.Module):
        def __init__(self):
            super().__init__()
            m = cfg['members']
            dims = [dim]+cfg['widths']+[1]
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            for j, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                w = torch.empty(m, a, b, device=device).normal_(0, np.sqrt(2/a))
                if j == len(dims)-2:
                    w.zero_()
                self.weights.append(nn.Parameter(w))
                self.biases.append(nn.Parameter(torch.zeros(m, b, device=device)))

        def forward(self, x):
            x = x[:, None, :].expand(-1, cfg['members'], -1)
            for j, (w, b) in enumerate(zip(self.weights, self.biases)):
                x = torch.einsum('bki,kio->bko', x, w)+b
                if j < len(self.weights)-1:
                    x = F.gelu(x)
                    x = F.dropout(x, cfg['dropout'], self.training)
            return x.squeeze(-1)
    return Ensemble()


def fit_predict(X, Q, types, fit_y, mf, mq, teacher_q, cfg, seed):
    """Returns the query-row margins of every member (len(Q) x members)."""
    import torch
    from torch.nn import functional as F
    torch.backends.cuda.matmul.allow_tf32 = True
    assert len(fit_y) == len(X) and len(teacher_q) == len(Q)
    X = np.column_stack([dense(X, types), mf])
    Q = np.column_stack([dense(Q, types), mq])
    mean = X.mean(0, dtype='float64')
    scale = np.maximum(X.std(0, dtype='float64'), .001)
    X = np.clip((X-mean)/scale, -10, 10).astype('float32')
    Q = np.clip((Q-mean)/scale, -10, 10).astype('float32')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = 'cuda'
    net = make_model(X.shape[1], cfg, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    xt = torch.tensor(X, device=dev)
    qt = torch.tensor(Q, device=dev)
    yt = torch.tensor(fit_y, device=dev, dtype=torch.float32)
    st = torch.tensor(teacher_q, device=dev, dtype=torch.float32)
    mt = torch.tensor(mf, device=dev)
    qm = torch.tensor(mq, device=dev)
    batch = cfg['batch']
    steps = cfg['epochs']*int(np.ceil(len(X)/batch))
    step = 0
    for epoch in range(cfg['epochs']):
        net.train()
        order = torch.randperm(len(X), device=dev)
        uorder = torch.randperm(len(Q), device=dev)
        total = 0.
        for j in range(0, len(X), batch):
            ix = order[j:j+batch]
            ux = uorder[torch.arange(j, j+len(ix), device=dev) % len(Q)]
            a = xt[ix]
            b = qt[ux]
            a = a+cfg['noise']*torch.randn_like(a)
            b = b+cfg['noise']*torch.randn_like(b)
            zz = net(torch.cat([a, b], 0))
            z = mt[ix, None]+zz[:len(ix)]
            uz = qm[ux, None]+zz[len(ix):]
            truth = F.binary_cross_entropy_with_logits(z, yt[ix, None].expand_as(z))
            soft = F.binary_cross_entropy_with_logits(uz, st[ux, None].expand_as(uz))
            loss = truth+cfg['distill_weight']*soft
            lr = cfg['lr']*(.02+.98*.5*(1+np.cos(np.pi*step/max(steps-1, 1))))
            for g in opt.param_groups:
                g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            step += 1
            total += float(loss.detach())*len(ix)
        print(f'  epoch {epoch}: loss {total/len(X):.5f}', flush=True)
    net.eval()
    margins = []
    with torch.no_grad():
        for j in range(0, len(Q), 4096):
            z = qm[j:j+4096, None]+net(qt[j:j+4096])
            margins.append(z.cpu().numpy())
    return np.concatenate(margins)
