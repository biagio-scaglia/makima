"""Distribuzioni probabilistiche analitiche per la sperimentazione in Python."""

import math


class Bernoulli:
    """Distribuzione di Bernoulli in Python per prototipazione rapida."""

    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p deve essere in [0.0, 1.0], ricevuto: {p}")
        self.p = float(p)
        self.q = 1.0 - self.p

    @property
    def mean(self) -> float:
        return self.p

    @property
    def variance(self) -> float:
        return self.p * self.q

    @property
    def entropy_bits(self) -> float:
        term_p = self.p * math.log2(self.p) if self.p > 0 else 0.0
        term_q = self.q * math.log2(self.q) if self.q > 0 else 0.0
        return -(term_p + term_q)

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p:.4f}, entropy={self.entropy_bits:.4f} bits)"


class BetaDistribution:
    """Distribuzione Beta e prior coniugato bayesiano per il laboratorio."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha e beta devono essere strettamente positivi (> 0.0)")
        self.alpha = float(alpha)
        self.beta = float(beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> float | None:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1.0))

    def bayesian_update(self, successes: int, failures: int) -> "BetaDistribution":
        """Aggiorna il prior osservando successi e fallimenti."""
        return BetaDistribution(self.alpha + successes, self.beta + failures)

    def ascii_density(self, width: int = 40) -> str:
        """Genera una rappresentazione testuale ASCII della stima di probabilità."""
        m = self.mean
        pos = int(round(m * (width - 1)))
        bar = ["-"] * width
        bar[pos] = "*"
        return f"[{''.join(bar)}] E[P]={m:.3f} (alpha={self.alpha}, beta={self.beta})"

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha:.2f}, beta={self.beta:.2f}, mean={self.mean:.4f}, var={self.variance:.4f})"


class PoissonDistribution:
    """Distribuzione di Poisson per la modellazione dei conteggi temporali."""

    def __init__(self, lambda_param: float):
        if lambda_param <= 0:
            raise ValueError("lambda deve essere strettamente positivo (> 0.0)")
        self.lambda_param = float(lambda_param)

    @property
    def mean(self) -> float:
        return self.lambda_param

    @property
    def variance(self) -> float:
        return self.lambda_param

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        # log-scale per stabilità
        log_pmf = k * math.log(self.lambda_param) - self.lambda_param - math.lgamma(k + 1)
        return math.exp(log_pmf)

    def __repr__(self) -> str:
        return f"Poisson(lambda={self.lambda_param:.2f}, mean={self.mean:.2f})"
