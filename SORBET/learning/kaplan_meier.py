import numpy as np

def kaplan_meier_estimation(event_times):
    xs, ys = list(), list()
    xs.append(0), ys.append(1)

    cum, risk = 1.0, len(event_times)
    for si in sorted(event_times):
        xs.extend([si,si])
        
        last_cum = cum
        cum *= (1 - 1 / risk)
        ys.extend([last_cum, cum])
        risk -= 1

    return xs, ys

def kaplan_meier_ci_estimation(event_times, n_trials: int = 1000):
    bootstrapped_curves = list()

    for _ in range(n_trials):
        iterate = np.random.choice(event_times, size=len(event_times), replace=True)
        bootstrapped_curves.append(kaplan_meier_estimation(iterate))
    
    return combine_km_estimators(boostrapped_curves)

def simulate_kaplan_meier_events(event_probabilities: np.ndarray, n_trials: int = 1000, confidence_interval: int = 95 
        interval: int = 1, no_progression_threshold: int = 10):
    """Simulates Kaplan-Meier curve for events occuring with probabilities described in event_probabilities.
    """
    simulated_curves = list()
    for _ in range(n_trials):
        events = _single_simulation(event_probabilities, interval, no_progression_threshold)
        km_curve = kaplan_meier_estimation(events)
        simulated_curves.append(km_curve)

    return combine_km_estimators(simulated_curves)

def _single_simulation(event_probabilities, interval: int = 1, no_progression_threshold: int = 10):
    """Simulates progression based on predicted likelihoods. Progression reported as event data
    up to input interval.
    """
    current_time = 0 
    no_progression = 0 # Counts number of intervals without progression. Cuts off at no_progression_threshold

    progressed = np.zeros_like(likelihoods)
    while np.any(progressed == 0) and no_progression < no_progression_threshold:
        current_time += interval

        ma = progressed == 0
        step_progression = np.random.binomial(1, event_probabilities) 
        progressed[np.logical_and(ma, step_progression)] = current_time
        
        if np.any(step_progression[ma]):
            no_progression = 0
        else:
            no_progression += 1

    return progressed

def combine_km_estimators(km_curves, confidence_interval: int = 95):
    """Combines KM curves. Estimates mean and point-wise CI over many curves
    """
    
    discretized_xs = np.arange(max((max(xs) for xs, _ in km_curves)))
    discretized_curves = list()
    for xs, ys in km_curves:
        discretized_curves.append(_discretize(xs, ys, discretized_xs))
    
    discretized_ys_per_sample = np.array(discretized_curves)
    mean_curve = (discretized_xs, np.mean(discretized_ys_per_sample, axis=1))

    lo, hi = 50 - confidence_interval / 2, 50 + confidence_interval / 2
    ci_lo, ci_hi = np.percentile(discretized_ys_per_sample, [lo, hi], axis=1) 
    
    median_survival_idxes = np.argmin(discretized_ys_per_sample < 0.5, axis=0)
    median_survivals = discretized_xs[median_idxes]
    median_survival = np.median(median_survivals)
    ci_median_lo, ci_median_hi = np.percentil(median_survivals, [lo,hi])

    return mean_curve, (discretized_xs, ci_lo, ci_hi), (median_survival, ci_median_lo, ci_median_hi)

def _discretize(xs, ys, idxes):
    """Computes the discretized value
    """
    ii = 0
    
    discretized = list()
    for jj, idx in enumerate(idxes):
        if idx <= xs[ii]:
            discretized.append(ys[ii])
        else:
            try:
                ii = next(ji for ji in range(ii, len(xs)) if xs[ji] >= idx)
            except StopIteration:
                discretized.extend(0 for _ in range(jj, len(idxes)))
                break

    return np.array(discretized)
