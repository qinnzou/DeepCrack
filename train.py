from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset
from tqdm import tqdm
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main():
    # ----------------------- dataset ----------------------- #

    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    train_pipline = dataReadPip(transforms=data_augment_op)

    test_pipline = dataReadPip(transforms=None)

    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    test_dataset = loadedDataset(readIndex(cfg.test_data_path), preprocess=test_pipline)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=4, drop_last=True)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')

    try:

        for epoch in range(1, cfg.epoch):
            trainer.vis.log('Start Epoch %d ...' % epoch, 'train info')
            model.train()

            # ---------------------  training ------------------- #
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            for idx, (img, lab) in bar:
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                pred = trainer.train_op(data, target)
                if idx % cfg.vis_train_loss_every == 0:
                    trainer.vis.log(trainer.log_loss, 'train_loss')
                    trainer.vis.plot_many({
                        'train_total_loss': trainer.log_loss['total_loss'],
                        'train_output_loss': trainer.log_loss['output_loss'],
                        'train_fuse5_loss': trainer.log_loss['fuse5_loss'],
                        'train_fuse4_loss': trainer.log_loss['fuse4_loss'],
                        'train_fuse3_loss': trainer.log_loss['fuse3_loss'],
                        'train_fuse2_loss': trainer.log_loss['fuse2_loss'],
                        'train_fuse1_loss': trainer.log_loss['fuse1_loss'],
                    })

                if idx % cfg.vis_train_acc_every == 0:
                    trainer.acc_op(pred[0], target)
                    trainer.vis.log(trainer.log_acc, 'train_acc')
                    trainer.vis.plot_many({
                        'train_mask_acc': trainer.log_acc['mask_acc'],
                        'train_mask_pos_acc': trainer.log_acc['mask_pos_acc'],
                        'train_mask_neg_acc': trainer.log_acc['mask_neg_acc'],
                    })
                if idx % cfg.vis_train_img_every == 0:
                    trainer.vis.img_many({
                        'train_img': data.cpu(),
                        'train_output': torch.sigmoid(pred[0].contiguous().cpu()),
                        'train_lab': target.unsqueeze(1).cpu(),
                        'train_fuse5': torch.sigmoid(pred[1].contiguous().cpu()),
                        'train_fuse4': torch.sigmoid(pred[2].contiguous().cpu()),
                        'train_fuse3': torch.sigmoid(pred[3].contiguous().cpu()),
                        'train_fuse2': torch.sigmoid(pred[4].contiguous().cpu()),
                        'train_fuse1': torch.sigmoid(pred[5].contiguous().cpu()),
                    })

                if idx % cfg.val_every == 0:
                    trainer.vis.log('Start Val %d ....' % idx, 'train info')
                    # -------------------- val ------------------- #
                    model.eval()
                    val_loss = {
                        'eval_total_loss': 0,
                        'eval_output_loss': 0,
                        'eval_fuse5_loss': 0,
                        'eval_fuse4_loss': 0,
                        'eval_fuse3_loss': 0,
                        'eval_fuse2_loss': 0,
                        'eval_fuse1_loss': 0,
                    }
                    val_acc = {
                        'mask_acc': 0,
                        'mask_pos_acc': 0,
                        'mask_neg_acc': 0,
                    }

                    bar.set_description('Epoch %d --- Evaluation --- :' % epoch)

                    with torch.no_grad():
                        for idx, (img, lab) in enumerate(val_loader, start=1):
                            val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(
                                torch.cuda.FloatTensor).to(device)
                            val_pred = trainer.val_op(val_data, val_target)
                            trainer.acc_op(val_pred[0], val_target)
                            val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                            val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                            val_loss['eval_fuse5_loss'] += trainer.log_loss['fuse5_loss']
                            val_loss['eval_fuse4_loss'] += trainer.log_loss['fuse4_loss']
                            val_loss['eval_fuse3_loss'] += trainer.log_loss['fuse3_loss']
                            val_loss['eval_fuse2_loss'] += trainer.log_loss['fuse2_loss']
                            val_loss['eval_fuse1_loss'] += trainer.log_loss['fuse1_loss']
                            val_acc['mask_acc'] += trainer.log_acc['mask_acc']
                            val_acc['mask_pos_acc'] += trainer.log_acc['mask_pos_acc']
                            val_acc['mask_neg_acc'] += trainer.log_acc['mask_neg_acc']
                        else:
                            trainer.vis.img_many({
                                'eval_img': val_data.cpu(),
                                'eval_output': torch.sigmoid(val_pred[0].contiguous().cpu()),
                                'eval_lab': val_target.unsqueeze(1).cpu(),
                                'eval_fuse5': torch.sigmoid(val_pred[1].contiguous().cpu()),
                                'eval_fuse4': torch.sigmoid(val_pred[2].contiguous().cpu()),
                                'eval_fuse3': torch.sigmoid(val_pred[3].contiguous().cpu()),
                                'eval_fuse2': torch.sigmoid(val_pred[4].contiguous().cpu()),
                                'eval_fuse1': torch.sigmoid(val_pred[5].contiguous().cpu()),

                            })
                            trainer.vis.plot_many({
                                'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                'eval_output_loss': val_loss['eval_output_loss'] / idx,
                                'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,

                            })
                            trainer.vis.plot_many({
                                'eval_mask_acc': val_acc['mask_acc'] / idx,
                                'eval_mask_neg_acc': val_acc['mask_neg_acc'] / idx,
                                'eval_mask_pos_acc': val_acc['mask_pos_acc'] / idx,

                            })
                            # ----------------- save model ---------------- #
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx) and cfg.save_acc < (
                                    val_acc['mask_acc'] / idx):
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                trainer.saver.save(model, tag='%s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc))
                                trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc), 'train info')

                    bar.set_description('Epoch %d --- Training --- :' % epoch)
                    model.train()

            if epoch != 0:
                trainer.saver.save(model, tag='%s_epoch(%d)' % (
                    cfg.name, epoch))
                trainer.vis.log('Save Model -%s_epoch(%d)' % (
                    cfg.name, epoch), 'train info')

    except KeyboardInterrupt:

        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        trainer.vis.log('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name,
                        'train info')
        trainer.vis.log('Training End!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
