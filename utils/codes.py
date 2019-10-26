def properties(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != '_']

class Models:
    AE= 'AE'
    VAE= 'VAE'
    BVAE = 'BVAE'
    AnnVAE = 'AnnVAE'
    AnnBVAE = 'AnnBVAE'
    AnnBayAE = 'AnnBayAE'
    AnnBayVAE = 'AnnBayVAE'
    AnnBayBVAE = 'AnnBayBVAE'
    DIPcovVAE = 'DIPcovVAE'
    DIPcovAE = 'DIPcovAE'
    DIPgaussVAE = 'DIPgaussVAE'
    DIPgaussAE = 'DIPgaussAE'
    BTCVAE = 'BTCVAE'
    BayAE = 'BayAE'
    BayVAE = 'BayVAE'
    Embed = 'Embed'



class Losses:
    OLS = 'OLS'
    MLE = 'MLE'

    KLD = 'KLD'   #KL-divergence
    RKLD = 'RKLD'  # Reverse KL-divergence
    JS = 'JS' #Jensen-Shannon
    CHI2 = 'CHI2'  # Chi-Square
    Helling = 'Helling'  # hellinger

    def get_label(self, mode):
        if   mode == self.OLS:
            return 'Ordinary Least Squares'
        elif mode == self.MLE:
            return 'Maximum Likelihood Estimation'

        elif mode == self.KLD:
            return 'Kullback-Leibler'
        elif mode == self.RKLD:
            return 'Reverse Kullback-Leibler'
        elif mode == self.JS:
            return 'Jensen-Shannon'
        elif mode == self.CHI2:
            return 'Chi-Square'
        elif mode == self.Helling:
            return 'Hellinger'


class Kernels:
    StudentT = 'StudentT'
    Cauchy = 'Cauchy'
    RBF = 'RBF'



