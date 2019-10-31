import gc
import os

import numpy as np
import tensorflow as tf
import dask.array as da
from dask_ml.decomposition import PCA

from tqdm import tqdm as tqdm
from collections import defaultdict

from utils.plots import plot_dataset, plot_dataset3d, plot_samples, merge

from skimage.transform import resize
import matplotlib.pyplot as plt

class BaseModel:

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, saver, global_step_tensor):
        print("Saving model...")
        saver.save(sess, self.config.checkpoint_dir + '/', global_step_tensor)
        print("Model saved" )

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, saver):
        retval = False
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir+ '/')

        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            retval = True
        else:
            print("Model does NOT exist")

        return retval

    def _train(self):
        raise NotImplementedError

    def _test(self):
        raise NotImplementedError


    '''  
    ------------------------------------------------------------------------------
                                         MODEL OPERATIONS
    ------------------------------------------------------------------------------ 
    '''
    def reconst(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        '''
        ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------
        '''
        inputs = inputs/255
        return self.batch_function(self.model_graph.encode, inputs)

    def decode(self, latent):
        return self.batch_function(self.model_graph.decode, latent)

    def reconst_loss(self, inputs):
        inputs = inputs/255
        return self.batch_function(self.model_graph.reconst_loss, inputs)

    def interpolate(self, input1, input2):
        z1 = self.encode(input1)
        z2 = self.encode(input2)

        decodes = defaultdict(list)
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            decode = dict()
            z = np.stack([self.slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.decode(z)*255

            for i in range(z_decode.shape[0]):
                try:
                    decode[i] = [z_decode[i].compute()]
                except:
                    decode[i] = [z_decode[i]]

            for i in range(z_decode.shape[0]):
                decodes[i] = decodes[i] + decode[i]

        imgs = []

        for idx in decodes:
            l = []

            l += [input1[idx:idx + 1][0]]
            l += decodes[idx]
            l += [input2[idx:idx + 1][0]]

            imgs.append(l)
        del decodes

        return imgs

    def slerp(self, val, low, high):
        """Code from https://github.com/soumith/dcgan.torch/issues/14"""
        omega = da.arccos(da.clip(da.dot(low / da.linalg.norm(low), high.transpose() / da.linalg.norm(high)), -1, 1))
        so = da.sin(omega)

        if so == 0:
            return (1.0-val) * low + val * high # L'Hopital's rule/LERP
        return da.sin((1.0 - val) * omega) / so * low + da.sin(val * omega) / so * high

    ''' 
    ------------------------------------------------------------------------------
                                         MODEL FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''
    def plot_sampling_reconst(self, std_scales, plot_scale=10, bn='', random_latent=None, save=True):

        assert len(std_scales) == self.config.latent_dim, 'The Standard Deviation\'s scales should be the same size as the latent dimension {}'\
            .format(self.config.latent_dim)
        print('=====================================================================')
        print('std scale: {}'.format(std_scales))

        samples = self._sampling_reconst(std_scales=std_scales, random_latent=random_latent).compute()

        im = merge(samples, (10, 10))
        fig_width = int(im.shape[0] * plot_scale)
        fig_height = int(im.shape[1] * plot_scale)
        im = resize(im, (fig_width, fig_height), anti_aliasing=True)
        plt.figure(dpi=150)
        plt.axis('off')
        pretoss = '' if random_latent is None else 'pretoss'
        if random_latent is None:
            dir_save = self.config.log_dir + '/total_random/{} sample{}.jpg'.format(self.config.log_dir.split('/')[-1:][0], bn)
        else:
            dir_save = self.config.log_dir + '/pretoss_random/{} {} sample{}.jpg'.format(self.config.log_dir.split('/')[-1:][0], pretoss, bn)

        print('Saving Image {} ...'.format(dir_save))
        plt.title('{}_{}'.format(pretoss, bn), loc='left')
        plt.imshow(im)

        if save:
            plt.savefig(dir_save)
            plt.close()
        else:
            plt.show()

    def plot_latent(self, cur_epoch=''):
        # Generating latent space
        if self.latent_data is None:
            self.generate_latent(self.data_train, self.session, self.config.ntrain_batches)

        pca = PCA(n_components=2)
        latent_pca = self.latent_data['latent'] if self.latent_data['latent'].shape[1] == 2 \
                                    else self.latent_data['latent'][:, 0:2] if self.latent_data['latent'].shape[1]==3 \
                                                                else pca.fit_transform(self.latent_data['latent'])

        print('Latent space dimensions: {}'.format(latent_pca.shape))
        print('Plotting latent space ...')
        latent_space = self.config.log_dir + '/latent2d/{} latent epoch {}.jpg'.format(self.config.log_dir.split('/')[-1:][0],
                                                                          cur_epoch)
        plot_dataset(latent_pca.compute(), y=self.latent_data['label'][:, self.latent_data['y_index']].compute(), save=latent_space)

        if self.latent_data['latent'].shape[1] >= 3:
            pca = PCA(n_components=3)
            latent_pca = self.latent_data['latent'] if self.latent_data['latent'].shape[1]==3 \
                                        else pca.fit_transform(self.latent_data['latent'])

            print('latent space dimensions: {}'.format(latent_pca.shape))
            print('Plotting latent space ...')
            latent_space = self.config.log_dir + '/latent3d/{} latent_3d epoch {}.jpg'.format(self.config.log_dir.split('/')[-1:][0],
                                                                                 cur_epoch)
            plot_dataset3d(latent_pca.compute(), y=self.latent_data['label'][:, self.latent_data['y_index']].compute(), save=latent_space)

        del latent_pca
        gc.collect()

    def plot_reconst(self, cur_epoch=''):
        # Generating Samples
        print('Reconstructing samples from Data ...')
        x_recons_l = self.reconst(self.samples)
        recons_file = self.config.log_dir + '/reconst/{} samples tsne_cost epoch {}.jpg'.format(
            self.config.log_dir.split('/')[-1:][0], cur_epoch)
        plot_samples(x_recons_l, scale=10, save=recons_file)

        del x_recons_l
        gc.collect()

    ''' 
    ------------------------------------------------------------------------------
                                         MODEL FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''
    def decay_fn(self):
        return self.model_graph.decay_lr(self.session)

    def batch_function(self, func, p1):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if (self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return

            output_l = list()

            start = 0
            end = self.config.batch_size

            with tqdm(range(p1.shape[0] // self.config.batch_size)) as pbar:
                while end < p1.shape[0]:
                    output = func(session, p1[start:end])
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))
                    output_l.append(output)

                    start = end
                    end += self.config.batch_size
                    pbar.update(1)
                else:

                    x1 = p1[start:]
                    xsize = len(x1)
                    p1t = da.ones([self.config.batch_size - xsize] + list(x1.shape[1:]))*-1

                    output = func(session, np.concatenate((x1, p1t), axis=0))
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))[0:xsize]

                    output_l.append(output)

                    pbar.update(1)

        try:
            return da.vstack(output_l)
        except:
            output_l = list(map(lambda l: l.reshape(-1, 1), output_l
                                )
                            )
        return da.vstack(output_l)

    ''' 
    ------------------------------------------------------------------------------
                                         COLAB FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''
    def zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def zipExperiments(self):
        import zipfile as zf
        zipf = zf.ZipFile(self.config.model_name+'.zip', 'w', zf.ZIP_DEFLATED)
        self.zipdir(self.experiments_root_dir+'/', zipf)
        zipf.close()


    def upzipExperiments(self):
        import zipfile as zf
        with zf.ZipFile(self.config.model_name+'.zip', 'r', zf.ZIP_DEFLATED) as zipf:
            zipf.extractall(self.experiments_root_dir+'/')
            zipf.close()

    def push_colab(self):
        self.zipExperiments()
        self.colab2google()

    def google2colab(self):
        from google.colab import auth
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.discovery import build

        file_name = self.config.model_name+'.zip'
        print('zip experiments {} ...'.format(file_name))
        file_path = './'+ file_name

        auth.authenticate_user()
        drive_service = build('drive', 'v3')
        files = drive_service.files().list().execute()
        for f in files['files']:
            if f['name'] == file_path:
                request = f['mimeType'].files().get_media(fileId=f['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print("Download %d%%." % int(status.progress() * 100))
                return fh.getvalue()
                break

    def colab2google(self):
        from google.colab import auth
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build


        file_name = self.config.model_name+'.zip'
        print('zip experiments {} ...'.format(file_name))
        file_path = './'+ file_name

        auth.authenticate_user()
        drive_service = build('drive', 'v3')

        print('uploading to google drive ...')
        file_metadata = {
            'name': file_name,
            'mimeType': 'application/octet-stream',
            'parents': [self.config.colabpath]
        }
        media = MediaFileUpload(file_path,
                                mimetype='application/octet-stream',
                                resumable=True)
        created = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print('File ID: {}'.format(created.get('id')))
