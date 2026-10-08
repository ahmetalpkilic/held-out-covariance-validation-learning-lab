import unittest
import numpy as np
from scipy.stats import f
import lab

class ValidationTests(unittest.TestCase):
    def test_scatter_and_event_offsets(self):
        rng=np.random.default_rng(1);z=rng.normal(size=(40,4,2));r=lab.validate_classes(z)
        np.testing.assert_allclose(lab.validate_classes(z+np.arange(40)[:,None,None]*7)['ratios'],r['ratios'],rtol=1e-13)
        # Independent NumPy covariance per event, then projection of each matrix.
        cov=np.stack([np.cov(v,rowvar=False) for v in z]);ratios=[]
        for g in (0,1):
            a,b=[cov[(lab.CLASSES==g)&(lab.CONDITIONS==h)].mean(axis=0) for h in (0,1)]
            ratios.extend([v@b@v/(v@a@v) for v in lab.PROJECTIONS])
        np.testing.assert_allclose(r['ratios'],ratios,rtol=1e-13)
        np.testing.assert_allclose(lab.validate_classes(z*1000)['p_values'],r['p_values'],atol=1e-13)

    def test_exact_null_and_power_independent_chisquare(self):
        rng=np.random.default_rng(42)
        for m in (2,8):
            df=10*(m-1);ratio=rng.chisquare(df,200000)/rng.chisquare(df,200000)
            lo,hi=f.ppf([lab.ALPHA/12,1-lab.ALPHA/12],df,df)
            self.assertAlmostEqual(lab.marginal_power(1,m),lab.ALPHA/6,places=12)
            for q in (1,1.25,2):
                power=lab.marginal_power(q,m);observed=np.mean((q*q*ratio<lo)|(q*q*ratio>hi))
                self.assertLess(abs(observed-power),6*np.sqrt(max(power*(1-power),1e-5)/len(ratio)))

    def test_family_false_alarms_and_power_with_raw_repeats(self):
        rng=np.random.default_rng(45);trials=1800;counts={1:0,2:0}
        for _ in range(trials):
            z=rng.normal(size=(40,4,2));z[:,:,1]=.5*z[:,:,0]+np.sqrt(.75)*z[:,:,1]
            for q in counts:
                scaled=z*np.where(lab.CONDITIONS,q,1)[:,None,None]
                counts[q]+=lab.validate_classes(scaled)['reject']
        self.assertLess(counts[1]/trials,.05+4*np.sqrt(.05*.95/trials))
        self.assertGreater(counts[2]/trials,.95)

    def test_independent_seed_selection_tradeoff(self):
        r=lab.experiment(trials=100,seed=431)
        for row in r['rows']:
            if row['q']==1:self.assertLess(row['rejection_rate'],.13)
            if row['q']==4:
                self.assertGreater(row['rejection_rate'],.95)
                self.assertLess(row['methods']['selected']['rmse_ms'],row['methods']['always_pool']['rmse_ms'])
            for value in row['methods'].values():self.assertEqual(value['successful_trials'],100)

    def test_invalid(self):
        for z in [np.ones((40,1,2)),np.ones((40,4,2)),np.ones((39,4,2)),np.full((40,4,2),np.nan)]:
            with self.assertRaises(ValueError):lab.validate_classes(z)
        for q,m in [(0,4),(1,1),(1,2.5)]:
            with self.assertRaises(ValueError):lab.marginal_power(q,m)

if __name__=='__main__':unittest.main()
