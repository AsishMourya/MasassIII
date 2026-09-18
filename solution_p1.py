from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cho_factor, cho_solve
ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True,
                     'grid.alpha': .22, 'figure.dpi': 120})
RESULTS = {}

def graph(n, p, rng):
    # Rejection sampling: ER graph conditioned on connectedness.
    while True:
        g = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))
        if nx.is_connected(g):
            return g

def draw_graph(g, ax, title):
    nx.draw_networkx(g, nx.spring_layout(g, seed=9), ax=ax,
                     labels={i: i+1 for i in g}, node_color='#d3e9f6',
                     node_size=430, font_size=9, edge_color='#8c99a3')
    ax.set_title(title)
    ax.set_axis_off()

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), bbox_inches='tight')
    plt.close(fig)

# ======================= PROBLEM 1 =======================
def problem1():
    import imageio.v2 as imageio
    rng = np.random.default_rng(101)
    n, p = 20, .25
    g = graph(n, p, rng)
    L = nx.laplacian_matrix(g).toarray().astype(float)
    h = .9 / max(dict(g.degree()).values())
    X = rng.uniform(-2.5, 2.5, (n, 2))
    center = X.mean(0)
    def line(a, b, count):
        return np.linspace(a, b, count)
    # Exactly 20 distinct targets per letter.
    A = np.vstack([line((-1,-1),(0,1),8),
                   line((0,1),(1,-1),8)[1:],
                   line((-.45,-.1),(.45,-.1),5)])
    S = np.vstack([line((1,1),(-1,1),5),
                   line((-1,1),(-1,0),4)[1:],
                   line((-1,0),(1,0),5)[1:],
                   line((1,0),(1,-1),4)[1:],
                   line((1,-1),(-1,-1),6)[1:]])
    I = np.vstack([line((-1,1),(1,1),6),
                   line((0,.8),(0,-.8),8),
                   line((-1,-1),(1,-1),6)])
    H = np.vstack([line((-1,-1),(-1,1),8),
                   line((1,-1),(1,1),8),
                   line((-.6,0),(.6,0),4)])
    letters = dict(A=A, S=S, I=I, H=H)
    frames, labels, ends, errors, bounds = [], [], [], [], [0]
    for letter in 'ASISH':
        D = letters[letter]
        assert D.shape == (20,2) and len(np.unique(D,axis=0)) == 20
        D = D - D.mean(0) + center
        # Assign targets once at each switch, then keep labels fixed.
        _, order = linear_sum_assignment(((X[:,None]-D[None])**2).sum(2))
        D = D[order]
        for k in range(120):
            errors.append(float(np.linalg.norm(X-D)))
            if k % 2 == 0:
                frames.append(X.copy()); labels.append(letter)
            X = X - h * L @ (X-D)
        errors[-1]=float(np.linalg.norm(X-D))
        ends.append(X.copy()); bounds.append(len(errors))
        frames.extend([X.copy()]*15); labels.extend([letter]*15)
    fig, axes = plt.subplots(1,5,figsize=(12,2.8))
    for ax, Xend, letter in zip(axes, ends, 'ASISH'):
        ax.scatter(Xend[:,0],Xend[:,1],s=24)
        ax.set(title=letter, aspect='equal', xlim=(center[0]-1.5,center[0]+1.5),
               ylim=(center[1]-1.5,center[1]+1.5))
        ax.set_xlabel('x'); ax.set_ylabel('y')
    save(fig,'p1_letters')
    fig, ax = plt.subplots(1,2,figsize=(10,3.8))
    draw_graph(g,ax[0],'Connected ER graph: N=20, p=0.25')
    ax[1].semilogy(np.maximum(errors,1e-15))
    for b in bounds[1:-1]: ax[1].axvline(b,color='gray',ls=':',lw=1)
    ax[1].set(xlabel='Control step k',ylabel='Formation error (Frobenius norm)')
    save(fig,'p1_convergence')
    fig, ax = plt.subplots(figsize=(5.6,4.8))
    ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal',xlabel='x',ylabel='y')
    scatter = ax.scatter(frames[0][:,0],frames[0][:,1],s=45,c=np.arange(n),cmap='tab20',vmin=0,vmax=19)
    title = ax.set_title('')
    with imageio.get_writer(ROOT/'asish_formation.mp4',fps=20,
                           codec='libx264', macro_block_size=2) as writer:
        for frame, letter in zip(frames,labels):
            scatter.set_offsets(frame); title.set_text('ASISH: forming '+letter)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
    plt.close(fig)
    RESULTS['p1'] = dict(edges=g.number_of_edges(),h=h,
        lambda2=float(np.linalg.eigvalsh(L)[1]),
        final_errors=[float(errors[b-1]) for b in bounds[1:]],
        centroid=center.tolist())

if __name__=='__main__':
    for fun in [problem1]:
        print('Running',fun.__name__,flush=True); fun()
        (ROOT/'results.json').write_text(json.dumps(RESULTS,indent=2))
    print(json.dumps(RESULTS,indent=2))
