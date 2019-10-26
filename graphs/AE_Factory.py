
from graphs.Basic.AE_graph import AEGraph
from graphs.Basic.VAE_graph import VAEGraph
from graphs.Beta.BetaVAE_graph import BVAEGraph
from graphs.Annealed.AnnBetaVAE_graph import AnnBVAEGraph

from graphs.Beta.BetaTCVAE_graph import BTCVAEGraph
from graphs.Annealed.AnnVAE_graph import AnnVAEGraph
from graphs.DIP.DIPgaussVAE_graph import DIPgaussVAEGraph
from graphs.DIP.DIPgaussAE_graph import DIPgaussAEGraph
from graphs.DIP.DIPcovVAE_graph import DIPcovVAEGraph
from graphs.DIP.DIPcovAE_graph import DIPcovAEGraph
from graphs.Bayesian.BayAE_graph import BayAEGraph
from graphs.AnnealedBayesian.AnnBayAE_graph import AnnBayAEGraph
from graphs.Bayesian.BayVAE_graph import BayVAEGraph
from graphs.AnnealedBayesian.AnnBayBVAE_graph import AnnBayBVAEGraph

from graphs.Embed.Embedding_graph import EmbeddingGraph

from utils.codes import Models

def Factory(configuration):
    print('building {} graph ... '.format(configuration['graph_type']))

    if configuration['graph_type'] == Models.AE:
        return AEGraph(configuration)
    elif configuration['graph_type'] == Models.VAE:
        return VAEGraph(configuration)
    elif configuration['graph_type'] == Models.BTCVAE:
        return BTCVAEGraph(configuration)
    elif configuration['graph_type'] == Models.BVAE:
        return BVAEGraph(configuration)
    elif configuration['graph_type'] == Models.AnnBVAE:
        return AnnBVAEGraph(configuration)
    elif configuration['graph_type'] == Models.AnnVAE:
        return AnnVAEGraph(configuration)
    elif configuration['graph_type'] == Models.DIPcovVAE:
        return DIPcovVAEGraph(configuration)
    elif configuration['graph_type'] == Models.DIPgaussVAE:
        return DIPgaussVAEGraph(configuration)
    elif configuration['graph_type'] == Models.DIPcovAE:
        return DIPcovAEGraph(configuration)
    elif configuration['graph_type'] == Models.DIPgaussAE:
        return DIPgaussAEGraph(configuration)
    elif configuration['graph_type'] == Models.BayAE:
        return BayAEGraph(configuration)
    elif configuration['graph_type'] == Models.AnnBayAE:
        return AnnBayAEGraph(configuration)
    elif configuration['graph_type'] == Models.AnnBayBVAE:
        return AnnBayBVAEGraph(configuration)
    elif configuration['graph_type'] == Models.BayVAE:
        return BayVAEGraph(configuration)
    elif configuration['graph_type'] == Models.Embed:
        return EmbeddingGraph(configuration)