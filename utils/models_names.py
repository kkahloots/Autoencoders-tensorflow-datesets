from utils.codes import Models

def get_model_name_AE(model, config):
    conv = 'Conv'  if config.isConv else ''
    model_name = model + '_' \
                 + conv + '_' \
                 +  config.dataset_name+ '_' \
                 + 'lat' + str(config.latent_dim)  + '_' \
                 + 'h' + str(config.hidden_dim)    + '_' \
                 + 'lay' + str(config.num_layers)  + '_' \
                 + 'rect'+ str(config.reconst_loss)
    return model_name

def get_model_name_VAE(model, config):
    return get_model_name_AE(model, config) + '_' \
                 + 'div'+ str(config.div_cost)

def get_model_name(model, config):
    if model in [Models.AE]:
        return get_model_name_AE(model, config)

    elif model in [Models.VAE]:
        return get_model_name_VAE(model, config)

    elif model in [Models.BVAE, Models.BTCVAE]:
        return get_model_name_VAE(model, config) + '_' \
                 + 'b' + str(config.beta).replace('.','_')

    elif model in [Models.DIPcovAE, Models.DIPgaussAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'lmd' + str(config.lambda_d) + '_' \
                 + 'fact' + str(config.d_factor)

    elif model in [Models.DIPcovVAE, Models.DIPgaussVAE]:
        return get_model_name_VAE(model, config) + '_' \
                 + 'lmd' + str(config.lambda_d) + '_' \
                 + 'fact' + str(config.d_factor)

    elif model in [Models.AnnVAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'g' + str(config.ann_gamma)  + '_' \
                 + 'cmax' + str(config.c_max)   + '_' \
                 + 'ithd' + str(config.itr_thd)

    elif model in [Models.BayAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'mc' + str(config.MC_samples)

    elif model in [Models.BayVAE]:
        return get_model_name_VAE(model, config) + '_' \
                 + 'mc' + str(config.MC_samples)

    elif model in [Models.Embed]:
        conv = 'Conv' if config.isConv else ''
        model_name = model + '_' \
                     + conv + '_' \
                     + config.dataset_name + '_' \
                     + 'lat' + str(config.latent_dim) + '_' \
                     + 'h' + str(config.hidden_dim) + '_' \
                     + 'lay' + str(config.num_layers) + '_' \
                     + 'df' + str(config.df).replace('.', '_') + '_' \
                     + 'nCom' + str(config.n_components) + '_' \
                     + 'ker' + str(config.kernel_mode) + '_' \
                     + 'ediv' + str(config.e_div_cost)
        return model_name