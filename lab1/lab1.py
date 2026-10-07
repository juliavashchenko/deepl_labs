import argparse
import sys

import numpy as np
import torch
import sklearn.datasets

torch.set_default_dtype(torch.float64)


PER_CLASS_TRAIN = 35        
TOL_TORCH = 1e-12           
TOL_NUM = 1e-7             
H = 1e-6                    


def prepare_data():
    X_all, y_all =  sklearn.datasets.load_iris(return_X_y=True)
    X_all = X_all.astype(np.float64)

    gen = np.random.default_rng(0)
    tr_ids, te_ids = [], []
    for cls in (0, 1, 2):                               
        ids = np.flatnonzero(y_all == cls)
        gen.shuffle(ids)                                
        tr_ids += ids[:PER_CLASS_TRAIN].tolist()        
        te_ids += ids[PER_CLASS_TRAIN:].tolist()        

    X_tr, y_tr = X_all[tr_ids], y_all[tr_ids]
    X_te, y_te = X_all[te_ids], y_all[te_ids]


    x_mean = X_tr.mean(axis=0)
    S = X_tr.std(axis=0, ddof=0)
    return (X_tr - x_mean) / S, y_tr, (x_mean) / S, y_te


def init_layers():
    gen = np.random.default_rng(0)
    W1 = gen.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    W2 = gen.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return [(W1, np.zeros(8)), (W2, np.zeros(3))]



def stable_log_softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    s = logits - m
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def forward(layers, X, y):

    inputs, pre = [], []
    a = X
    last = len(layers) - 1
    for k, (W, b) in enumerate(layers):
        inputs.append(a)
        z = a @ W + b
        if k < last:
            pre.append(z)
            a = np.maximum(z, 0.0)          
        else:
            logits = z                      

    logp = stable_log_softmax(logits)
    loss = -logp[np.arange(len(y)), y].mean()
    cache = {"inputs": inputs, "pre": pre, "probs": np.exp(logp), "y": y,
             "weights": [W for W, _ in layers]}
    return loss, cache

def backward(cache, average=True):

    P, y = cache["probs"], cache["y"]
    n = P.shape[0]

    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    dz = P - onehot
    if average:
        dz = dz / n

    dWs, dbs = [], []
    for k in range(len(cache["weights"]) - 1, -1, -1):  
        dWs.append(cache["inputs"][k].T @ dz)
        dbs.append(dz.sum(axis=0))
        if k > 0:
            da = dz @ cache["weights"][k].T
            dz = da * (cache["pre"][k - 1] > 0.0)

    return dWs[::-1], dbs[::-1]      



def torch_loss_and_grads(layers, X, y):

    lin1, lin2 = torch.nn.Linear(4, 8), torch.nn.Linear(8, 3)
    with torch.no_grad():
        for lin, (W, b) in zip((lin1, lin2), layers):
            lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))
            lin.bias.copy_(torch.from_numpy(b.copy()))
    out = lin2(torch.relu(lin1(torch.from_numpy(X.copy()))))
    loss = torch.nn.functional.cross_entropy(out, torch.from_numpy(y.copy()))
    loss.backward()
    g = [lin1.weight.grad.numpy().T, lin1.bias.grad.numpy(),
         lin2.weight.grad.numpy().T, lin2.bias.grad.numpy()]
    return loss.item(), g


def print_table(header, rows):

    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    print(line(header))
    print("| " + " | ".join("-" * x for x in w) + " |")
    for r in rows:
        print(line(r))


def check_vs_torch(layers, X, y, average=True, verbose=True):
    loss_np, cache = forward(layers, X, y)
    dWs, dbs = backward(cache, average=average)
    mine = [dWs[0], dbs[0], dWs[1], dbs[1]]
    loss_t, theirs = torch_loss_and_grads(layers, X, y)

    items = [("Втрата", abs(loss_np - loss_t), np.isfinite([loss_np, loss_t]).all())]
    for name, a, b in zip(("W1", "b1", "W2", "b2"), mine, theirs):
        items.append((f"Градієнт {name}", np.abs(a - b).max(),
                      np.isfinite(a).all() and np.isfinite(b).all()))

    passed = [bool(fin and d <= TOL_TORCH) for _, d, fin in items]
    if verbose:
        print(f"Втрата NumPy:   {loss_np:.15f}")
        print(f"Втрата PyTorch: {loss_t:.15f}\n")
        print_table(["Величина", "Макс. абс. різниця NumPy / PyTorch", "Перевірку пройдено"],
                    [[n, f"{d:.3e}", "Так" if p else "Ні"] for (n, d, _), p in zip(items, passed)])
    return all(passed)


def check_numeric(layers, X, y, average=True, verbose=True):
    _, cache = forward(layers, X, y)
    dWs, dbs = backward(cache, average=average)


    targets = [("W1[0, 0]", layers[0][0], (0, 0), dWs[0][0, 0]),
               ("b1[0]",    layers[0][1], (0,),   dbs[0][0]),
               ("W2[0, 0]", layers[1][0], (0, 0), dWs[1][0, 0]),
               ("b2[0]",    layers[1][1], (0,),   dbs[1][0])]

    rows, passed = [], []
    for name, arr, idx, g_hand in targets:
        saved = arr[idx]                         
        arr[idx] = saved + H
        l_plus = forward(layers, X, y)[0]       
        arr[idx] = saved - H
        l_minus = forward(layers, X, y)[0]        
        arr[idx] = saved                          
        g_num = (l_plus - l_minus) / (2 * H)

        d = abs(g_num - g_hand)
        ok = bool(np.isfinite([g_num, g_hand]).all() and d <= TOL_NUM)
        passed.append(ok)
        rows.append([name, f"{g_hand:.10e}", f"{g_num:.10e}", f"{d:.3e}", "Так" if ok else "Ні"])

    if verbose:
        print_table(["Параметр", "Градієнт backward()", "Чисельна похідна",
                     "Абсолютна різниця", "Перевірку пройдено"], rows)
    return all(passed)


def run_normal(layers, X, y):
    loss, cache = forward(layers, X, y)
    dWs, dbs = backward(cache)
    print("Розмірності проміжних масивів:")
    print("  inputs:", [a.shape for a in cache["inputs"]],
          "| pre (Z1):", [z.shape for z in cache["pre"]], "| probs:", cache["probs"].shape)
    print("Розмірності градієнтів: dW1", dWs[0].shape, "db1", dbs[0].shape,
          "dW2", dWs[1].shape, "db2", dbs[1].shape, "\n")

    print("=== Звірка з PyTorch (|a-b| <= 1e-12) ===")
    ok1 = check_vs_torch(layers, X, y)
    print("\n=== Чисельна перевірка (eps=1e-6, |g_num-g| <= 1e-7) ===")
    ok2 = check_numeric(layers, X, y)
    print("\nПідсумок:", "усе пройдено" if ok1 and ok2 else "ЄСТЬ НЕПРОЙДЕНІ ПЕРЕВІРКИ")
    return ok1 and ok2


def run_experiment(layers, X, y):
    n = len(y)

    _, cache = forward(layers, X, y)
    good = backward(cache, average=True)
    bad = backward(cache, average=False)
    flat_good = [good[0][0], good[1][0], good[0][1], good[1][1]]
    flat_bad = [bad[0][0], bad[1][0], bad[0][1], bad[1][1]]
    for name, g, b in zip(("W1", "W2", "b1", "b2"), flat_good, flat_bad):
        mask = np.abs(g) > 1e-15
        r = b[mask] / g[mask]
        print(f"{name}: ненульових {mask.sum()}/{g.size}, відношення {r.min():.4f}..{r.max():.4f}")

    print("\n=== Звірка з PyTorch (з помилкою) ===")
    t_ok = check_vs_torch(layers, X, y, average=False)
    print("\n=== Чисельна перевірка (з помилкою) ===")
    n_ok = check_numeric(layers, X, y, average=False)

    detected = (not t_ok) or (not n_ok)
    print("\nПомилку виявлено:", "ТАК - прогноз підтверджено" if detected else "НІ - перевірте код")


    print("\n=== Відновлена правильна реалізація ===")
    back_ok = check_vs_torch(layers, X, y, verbose=False) and check_numeric(layers, X, y, verbose=False)
    print("Правильна реалізація проходить усі перевірки:", "так" if back_ok else "НІ")
    return detected and back_ok


def main():
    ap = argparse.ArgumentParser(description="ЛР1: ручний backprop та його перевірка")
    ap.add_argument("--experiment", action="store_true", help="дослід із навмисною помилкою")
    args = ap.parse_args()

    X_tr, y_tr, X_te, y_te = prepare_data()
    print(f"Навчальна: {X_tr.shape}, класи {np.bincount(y_tr)}; "
          f"тестова: {X_te.shape}, класи {np.bincount(y_te)}\n")

    layers = init_layers()    
    ok = run_experiment(layers, X_tr, y_tr) if args.experiment else run_normal(layers, X_tr, y_tr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()