import visdom
import numpy as np
import time
import torch
import os


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.env = env
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kwargs = kwargs

        self.index = {}
        self.log_text = {}

    def save_settings(self, save_path=None, save_log=False, save_img=False, save_plot=False):

        save_format = '{info}-{time}'.format(time=time.strftime("%Y-%m-%d %H:%M:%S"), info=self.env)
        self.save_path = os.path.join(save_path, save_format)

        if self.save_path and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.save_log = save_log

        if self.save_path and self.save_log:
            os.mkdir((os.path.join(self.save_path, 'logs')))

        self.save_img = save_img

        if self.save_path and self.save_img:
            os.mkdir((os.path.join(self.save_path, 'imgs')))
        self.save_plot = save_plot

        if self.save_path and self.save_plot:
            os.mkdir((os.path.join(self.save_path, 'plots')))

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def reinit(self, env, **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

    def state_dict(self):
        return {
            'index': self.index,
            'log_text': self.log_text,
            '_vis_kwargs': self._vis_kwargs,
            'env': self.vis.env,
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), use_incoming_socket=False, **d.get('_vis_kwargs'))
        self.log_text = d.get('log_text', {})
        self.index = d.get('index', {})

    def log(self, info, win='defalut'):

        """
        self.log({'loss':1,'lr':0.0001}, 'loss')
        self.log('start load dataset ...', 'info')
        self.log('acc TP:%f, FP:%f ....'%(a,b), 'acc')
        """

        if self.log_text.get(win, None) is not None:
            flag = True
        else:
            flag = False

        self.log_text[win] = ('[{time}] {info}\n'.format(time=time.strftime("%Y-%m-%d %H:%M:%S"), info=info))
        self.vis.text(self.log_text[win], win, append=flag)

        # if self.save_log:
        #     with open(os.path.join(self.save_path, 'logs', '%s.txt'%win), 'a') as f:
        #         f.write('%s'%(self.log_text[win]))

    def log_many(self, d):
        '''
        d: dict{'loss':{'loss':1,'lr':0.0001},
                'info':'start load dataset ...'
                'acc':'acc TP:%f, FP:%f ....'%(a,b)}
        '''

        for k, v in d.items():
            self.log(v, k)

    def img(self, img, win='default', **kwargs):
        '''
        only tensor or numpy
        self.img(t.Tensor(64,64))
        self.img(t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        '''

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        self.vis.images(img, win=win, opts=dict(title=win), **kwargs)

    def img_many(self, d):
        for k, v in d.items():
            self.img(v, k)

    def plot(self, y, win='loss', **kwargs):

        '''

        :param y: scale float
        :param win:
        :param kwargs:
        :return:
        '''
        x = self.index.get(win, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=win, opts=dict(title=win),

                      update=None if x == 0 else 'append', **kwargs)

        self.index[win] = x + 1

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(v, k)
