"""Raw-field embedding student: the student ensemble of distill_core.py reads the feature bank concatenated with a
128-d encoding of the 13 raw fields (16-d embeddings scaled by count reliability + standardized numerics + mask
indicators -> 256 -> 128). A masked-field reconstruction head on all train + test rows runs with weight aux_weight."""
import numpy as np

from distill_core import dense, make_model as make_classifier


def make_model(cards, priors, reliability, dim, cfg, device):
    import torch
    from torch import nn

    class Multitask(nn.Module):
        def __init__(self):
            super().__init__()
            nf = len(cards)
            embedding = cfg['aux_embedding']
            self.embeddings = nn.ModuleList([nn.Embedding(c, embedding) for c in cards])
            widths = [nf*embedding+2*nf]+cfg['aux_widths']
            layers = []
            for a, b in zip(widths[:-1], widths[1:]):
                layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(cfg['aux_dropout'])]
            self.encoder = nn.Sequential(*layers)
            self.heads = nn.ModuleList([nn.Linear(widths[-1], c) for c in cards])
            for j, head in enumerate(self.heads):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.register_buffer(f'prior_{j}', torch.as_tensor(priors[j], dtype=torch.float32))
                self.register_buffer(f'reliability_{j}', torch.as_tensor(reliability[j], dtype=torch.float32))
            self.classifier = make_classifier(dim+widths[-1], cfg, device)

        def encode(self, raw, num, masked=None):
            pieces = []
            for j, table in enumerate(self.embeddings):
                z = table(raw[:, j])*getattr(self, f'reliability_{j}')[raw[:, j], None]
                if j == masked:
                    z = z*0
                pieces.append(z)
            numeric = num.clone()
            indicator = torch.zeros_like(numeric)
            if masked is not None:
                numeric[:, masked] = 0
                indicator[:, masked] = 1
            return self.encoder(torch.cat(pieces+[numeric, indicator], dim=1))

        def classify(self, features, raw, num):
            return self.classifier(torch.cat([features, self.encode(raw, num)], dim=1))

        def reconstruct(self, raw, num, masked):
            return self.heads[masked](self.encode(raw, num, masked))+getattr(self, f'prior_{masked}')

    return Multitask().to(device)


def fit_predict(X, Q, types, mf, mq, raw, num, fit, query, fit_y, original, cards, cfg, seed):
    """Returns the member-mean margin of the query rows. raw / num: codes / values of the 13 fields of all rows."""
    import torch
    from torch.nn import functional as F
    torch.backends.cuda.matmul.allow_tf32 = True
    assert len(X) == len(fit) == len(fit_y) and len(Q) == len(query)
    X = np.column_stack([dense(X, types), mf])
    Q = np.column_stack([dense(Q, types), mq])
    mean = X.mean(0, dtype='float64')
    scale = np.maximum(X.std(0, dtype='float64'), .001)
    X = np.clip((X-mean)/scale, -10, 10).astype('float32')
    Q = np.clip((Q-mean)/scale, -10, 10).astype('float32')
    nmean = num[fit].mean(0, dtype='float64')
    nscale = np.maximum(num[fit].std(0, dtype='float64'), .001)
    numeric = np.clip((num-nmean)/nscale, -10, 10).astype('float32')
    priors, reliability = [], []
    for j, c in enumerate(cards):
        count = np.bincount(raw[:, j], minlength=c)
        prior = (count+20*original[j])/(len(raw)+20)
        priors.append(np.log(prior))
        reliability.append(np.sqrt(count/(count+20)))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = 'cuda'
    net = make_model(cards, priors, reliability, X.shape[1], cfg, device)
    xt = torch.tensor(X, device=device)
    qt = torch.tensor(Q, device=device)
    rt = torch.tensor(raw, device=device)
    nt = torch.tensor(numeric, device=device)
    ft = torch.tensor(fit, device=device)
    it = torch.tensor(query, device=device)
    yt = torch.tensor(fit_y, device=device, dtype=torch.float32)
    mt = torch.tensor(mf, device=device)
    qm = torch.tensor(mq, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    batch = cfg['batch']
    steps = cfg['epochs']*int(np.ceil(len(X)/batch))
    step = 0
    for epoch in range(cfg['epochs']):
        net.train()
        order = torch.randperm(len(X), device=device)
        uorder = torch.randperm(len(raw), device=device)
        true_total = aux_total = 0.
        for j in range(0, len(X), batch):
            ix = order[j:j+batch]
            global_ix = ft[ix]
            ux = uorder[j:j+len(ix)]
            features = xt[ix]+cfg['noise']*torch.randn_like(xt[ix])
            logits = mt[ix, None]+net.classify(features, rt[global_ix], nt[global_ix])
            true_loss = F.binary_cross_entropy_with_logits(logits, yt[ix, None].expand_as(logits))
            field = step % len(cards)
            aux_logits = net.reconstruct(rt[ux], nt[ux], field)
            aux_loss = F.cross_entropy(aux_logits, rt[ux, field])
            loss = true_loss+cfg['aux_weight']*aux_loss
            rate = cfg['lr']*(.02+.98*.5*(1+np.cos(np.pi*step/max(steps-1, 1))))
            for group in opt.param_groups:
                group['lr'] = rate
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            step += 1
            true_total += float(true_loss.detach())*len(ix)
            aux_total += float(aux_loss.detach())*len(ix)
        print(f'  epoch {epoch}: bce {true_total/len(X):.5f} aux {aux_total/len(X):.4f}', flush=True)
    net.eval()
    margins = []
    with torch.no_grad():
        for j in range(0, len(Q), 4096):
            qi = it[j:j+4096]
            z = qm[j:j+4096, None]+net.classify(qt[j:j+4096], rt[qi], nt[qi])
            margins.append(z.mean(1).cpu().numpy())
    return np.concatenate(margins)
