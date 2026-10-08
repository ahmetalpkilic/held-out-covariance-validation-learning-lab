"""Independent validation of prespecified covariance classes for optical timing."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import f
import optics as o
import clock_model as c
from pooling import CLASSES, covariances, estimate, summary

SEED = 20261008
ALPHA = .05
# Conditions are declared BEFORE observing repeats; their noise levels are unknown.
CONDITIONS = (np.arange(40)//2) % 2
PROJECTIONS = np.array([[1.,0.],[0.,1.],[1.,1.]])
LEVELS = (1.,1.25,1.5,2.,4.)
BUDGETS = (2,4,8)


def validate_classes(repeats, alpha=ALPHA):
    """Six two-sided F tests, Bonferroni familywise error <= alpha.

    Remove each event's OWN mean; ten events/condition/class contribute m-1
    degrees each. The projections are correlated; Bonferroni needs no
    independence among tests. Repeats and events must be independent Gaussian.
    """
    z = np.asarray(repeats,float)
    if z.ndim != 3 or z.shape[0] != 40 or z.shape[2] != 2 or z.shape[1] < 2 or not np.all(np.isfinite(z)):
        raise ValueError('Expected finite (40,m>=2,2) validation repeats')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1)')
    residual = z-z.mean(axis=1,keepdims=True)
    projected = residual @ PROJECTIONS.T
    df = 10*(z.shape[1]-1)
    ratios=[]
    for g in (0,1):
        variances=[]
        for h in (0,1):
            block=projected[(CLASSES==g)&(CONDITIONS==h)]
            variances.append(np.sum(block**2,axis=(0,1))/df)
        if np.any(np.asarray(variances)<=0):
            raise ValueError('Zero projected scatter')
        ratios.extend(variances[1]/variances[0])
    ratios=np.asarray(ratios)
    p=np.minimum(1.,2*np.minimum(f.cdf(ratios,df,df),f.sf(ratios,df,df)))
    return dict(reject=bool(np.any(p < alpha/6)),p_values=p,ratios=ratios,df=df)


def marginal_power(q,m,alpha=ALPHA):
    """Exact power of ONE of the six tests for covariance scaling q^2."""
    if not np.isfinite(q) or q<=0 or not isinstance(m,(int,np.integer)) or m<2 or not 0<alpha<1:
        raise ValueError('Require q>0, integer m>=2 and 0<alpha<1')
    df=10*(m-1);lo,hi=f.ppf([alpha/12,1-alpha/12],df,df)
    return float(f.cdf(lo/q**2,df,df)+f.sf(hi/q**2,df,df))


def experiment(trials=400,seed=SEED):
    if not isinstance(trials,int) or trials<2:raise ValueError('trials must be >=2')
    # Four disjoint random streams: optical image, evaluation, training, selection.
    image_rng,eval_rng,train_rng,select_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    truth=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
    center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers=np.array([o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,33)) for _ in range(trials)])
    evaluation=eval_rng.normal(size=(trials,40,2))
    training=train_rng.normal(size=(trials,40,4,2))
    selection=select_rng.normal(size=(trials,40,max(BUDGETS),2))
    rows=[]
    for q in LEVELS:
        cov=covariances(q);chol=np.linalg.cholesky(cov)
        # Shared evaluation and estimation data pair policy comparisons fairly.
        errors={k:[] for k in ('oracle','per_event','always_pool')}
        selected={m:[] for m in BUDGETS};rejects={m:[] for m in BUDGETS};marginal={m:[] for m in BUDGETS}
        for t,u in enumerate(centers):
            z=truth+np.einsum('nij,nj->ni',chol,evaluation[t])
            train=truth[:,None,:]+np.einsum('nij,nkj->nki',chol,training[t])
            assumed={'oracle':cov,'per_event':estimate(train),'always_pool':estimate(train,CLASSES)}
            # Fail loudly rather than quietly reporting performance conditional on success.
            e={k:1000*(c.delay_from_center(u,c.calibrate(*z.T,s))-c.DELAY) for k,s in assumed.items()}
            for k in errors:errors[k].append(e[k])
            heldout=truth[:,None,:]+np.einsum('nij,nkj->nki',chol,selection[t])
            for m in BUDGETS:
                check=validate_classes(heldout[:,:m])
                rejects[m].append(check['reject']);marginal[m].append(check['p_values'][0]<ALPHA/6)
                selected[m].append(e['per_event'] if check['reject'] else e['always_pool'])
        for m in BUDGETS:
            rate=float(np.mean(rejects[m]))
            rows.append(dict(q=q,selection_repeats=m,selection_pairs=40*m,df_per_test=10*(m-1),
                rejection_rate=rate,rejection_mcse=float(np.sqrt(rate*(1-rate)/trials)),
                marginal_rejection_rate=float(np.mean(marginal[m])),exact_marginal_power=marginal_power(q,m),
                methods={**{k:summary(v) for k,v in errors.items()},'selected':summary(selected[m])}))
    return dict(seed=seed,trials=trials,alpha_family=ALPHA,training_pairs=160,evaluation_pairs=40,
                sigma_px=o.SIGMA,center_sd_px=float(centers.std(ddof=1)),failed_fits=0,rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=experiment();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for m in BUDGETS:
        rows=[r for r in report['rows'] if r['selection_repeats']==m]
        axes[0].plot(LEVELS,[r['rejection_rate'] for r in rows],'o-',label=f'{m} selection repeats')
        axes[1].plot(LEVELS,[r['marginal_rejection_rate'] for r in rows],'o',label=f'm={m}: simulation')
        axes[1].plot(LEVELS,[r['exact_marginal_power'] for r in rows],'-',label=f'm={m}: exact')
        axes[2].plot(LEVELS,[r['methods']['selected']['rmse_ms'] for r in rows],'o-',label=f'Select: m={m}')
    for key,label in [('always_pool','Always pool'),('per_event','Per event'),('oracle','Known covariance')]:
        rows=[r for r in report['rows'] if r['selection_repeats']==4]
        axes[2].plot(LEVELS,[r['methods'][key]['rmse_ms'] for r in rows],'--',label=label)
    axes[0].axhline(ALPHA,color='k',linestyle=':',label='Null family bound 0.05')
    axes[0].set(ylabel='Probability of rejecting pooling',title='Six tests: false alarms and power',ylim=(-.03,1.05))
    axes[1].set(ylabel='Rejection probability, one test',title='Independent analytic power check',ylim=(-.03,1.05))
    axes[2].set(ylabel='Camera-delay RMSE (ms)',title='Selection has a measurement cost')
    for ax in axes:
        ax.set(xlabel='Within-class noise SD ratio q');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':demo()
