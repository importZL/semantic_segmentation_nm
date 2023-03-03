#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from unet.nnUNetTrainerV2 import nnUNetTrainerV2

import torch
from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem
from models_pix2pix import networks
import torchvision.transforms.functional as func


parser = argparse.ArgumentParser()
parser.add_argument("network")
parser.add_argument("network_trainer")
parser.add_argument('--gpu_ids', type=str, default='0', help='')
parser.add_argument("task", help="can be task name or task id")
parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                    action="store_true")
parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                    action="store_true")
parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                    default=default_plans_identifier, required=False)
parser.add_argument("--use_compressed_data", default=False, action="store_true",
                    help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                            "is much more CPU and RAM intensive and should only be used if you know what you are "
                            "doing", required=False)
parser.add_argument("--deterministic",
                    help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                            "this is not necessary. Deterministic training will make you overfit to some random seed. "
                            "Don't use that.",
                    required=False, default=False, action="store_true")
parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                        "export npz files of "
                                                                                        "predicted segmentations "
                                                                                        "in the validation as well. "
                                                                                        "This is needed to run the "
                                                                                        "ensembling step so unless "
                                                                                        "you are developing nnUNet "
                                                                                        "you should enable this")
parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                    help="not used here, just for fun")
parser.add_argument("--valbest", required=False, default=False, action="store_true",
                    help="hands off. This is not intended to be used")
parser.add_argument("--fp32", required=False, default=False, action="store_true",
                    help="disable mixed precision training and run old school fp32")
parser.add_argument("--val_folder", required=False, default="validation_raw",
                    help="name of the validation folder. No need to use this for most people")
parser.add_argument("--disable_saving", required=False, action='store_true',
                    help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                            "will be removed at the end of the training). Useful for development when you are "
                            "only interested in the results and want to save some disk space")
parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                    help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                            "closely observing the model performance on specific configurations. You do not need it "
                            "when applying nnU-Net because the postprocessing for this will be determined only once "
                            "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                            "running postprocessing on each fold is computationally cheap, but some users have "
                            "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                            "you should consider setting this flag.")
parser.add_argument("--disable_validation_inference", required=False, action="store_true",
                    help="If set nnU-Net will not run inference on the validation set. This is useful if you are "
                            "only interested in the test set results and want to save some disk space and time.")
# parser.add_argument("--interp_order", required=False, default=3, type=int,
#                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
# parser.add_argument("--interp_order_z", required=False, default=0, type=int,
#                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
#                          "Hands off")
# parser.add_argument("--force_separate_z", required=False, default="None", type=str,
#                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                    help='Validation does not overwrite existing segmentations')
parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                    help='do not predict next stage')
parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                    help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                            'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                            'Optional. Beta. Use with caution.')

##### Parameters for end2end training #####
parser.add_argument('--train_end2end', type=bool, default=True, help='')
parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")
# visualization parameters
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--display_freq', type=int, default=280*50, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=280*50, help='frequency of showing training results on console')
# network saving and loading parameters
parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--lambda_L1', type=float, default=1.0, help='')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=20, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--unet_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
# model parameters
parser.add_argument('--isTrain', type=str, default='True')
parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

args = parser.parse_args()

device = torch.device('cuda:0' if int(args.gpu_ids) == 0 else 'cuda:1')
task = args.task
fold = args.fold
network = args.network
network_trainer = args.network_trainer
validation_only = args.validation_only
plans_identifier = args.p
find_lr = args.find_lr
disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

use_compressed_data = args.use_compressed_data
decompress_data = not use_compressed_data

deterministic = args.deterministic
valbest = args.valbest

fp32 = args.fp32
run_mixed_precision = not fp32

val_folder = args.val_folder
# interp_order = args.interp_order
# interp_order_z = args.interp_order_z
# force_separate_z = args.force_separate_z

if not task.startswith("Task"):
    task_id = int(task)
    task = convert_id_to_task_name(task_id)

if fold == 'all':
    pass
else:
    fold = int(fold)

# if force_separate_z == "None":
#     force_separate_z = None
# elif force_separate_z == "False":
#     force_separate_z = False
# elif force_separate_z == "True":
#     force_separate_z = True
# else:
#     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

if trainer_class is None:
    raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

if network == "3d_cascade_fullres":
    assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
        "If running 3d_cascade_fullres then your " \
        "trainer class must be derived from " \
        "nnUNetTrainerCascadeFullRes"
else:
    assert issubclass(trainer_class,
                        nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

trainer = nnUNetTrainerV2(plans_file, fold, args, output_folder=output_folder_name, dataset_directory=dataset_directory,
                        batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                        deterministic=deterministic,
                        fp16=run_mixed_precision)
if args.disable_saving:
    trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
    trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
    # self.best_val_eval_criterion_MA
    trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
    # the training chashes
    trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

trainer.initialize(not validation_only)

##### Defination related to End-to-End training #####
criterionGAN = networks.GANLoss(args.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()
class Generator(ImplicitProblem):
    def training_step(self, batch):
        data = batch['data']
        target = batch['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=int(args.gpu_ids))
            target = to_cuda(target, gpu_id=int(args.gpu_ids))

        train_data = torch.reshape(data[0:-2], (-1, data.size()[-2], data.size()[-1])).unsqueeze(1)
        train_target = torch.reshape(target[0][0:-2], (-1, target[0].size()[-2], target[0].size()[-1])).unsqueeze(1)
        real_mask = train_target.type(torch.cuda.FloatTensor).to(device)
        real_image = train_data.type(torch.cuda.FloatTensor).to(device)

        fake_image = self.module(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1)
        pred_fake = self.netD(fake_mask_image)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_image, real_image) * args.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        return loss_G


class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        data = batch['data']
        target = batch['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=int(args.gpu_ids))
            target = to_cuda(target, gpu_id=int(args.gpu_ids))
        train_data = torch.reshape(data[0:-2], (-1, data.size()[-2], data.size()[-1])).unsqueeze(1)
        train_target = torch.reshape(target[0][0:-2], (-1, target[0].size()[-2], target[0].size()[-1])).unsqueeze(1)
        real_mask = train_target.type(torch.cuda.FloatTensor).to(device)
        real_image = train_data.type(torch.cuda.FloatTensor).to(device)

        fake_image = self.netG(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.module(fake_mask_image.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_mask_image = torch.cat((real_mask, real_image), 1)
        pred_real = self.module(real_mask_image)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


train_losses_epoch = []
class Unet(ImplicitProblem):
    def training_step(self, batch):
        data = batch['data']
        target = batch['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=int(args.gpu_ids))
            target = to_cuda(target, gpu_id=int(args.gpu_ids))
        

        data, target[0], target[1], target[2] = data[0:-2], target[0][0:-2], target[1][0:-2], target[2][0:-2]

        aug_target = []
        aug_target.append(func.rotate(target[0].reshape(-1, target[0].size()[-2], target[0].size()[-1]), 8))
        aug_target.append(func.rotate(target[1].reshape(-1, target[1].size()[-2], target[1].size()[-1]), 8))
        aug_target.append(func.rotate(target[2].reshape(-1, target[2].size()[-2], target[2].size()[-1]), 8))

        aug_data = trainer.model.netG(aug_target[0].unsqueeze(1))

        aug_target[0] = torch.reshape(aug_target[0], (7, 1, 40, 56, 40))
        aug_target[1] = torch.reshape(aug_target[1], (7, 1, 20, 28, 20))
        aug_target[2] = torch.reshape(aug_target[2], (7, 1, 10, 14, 10))
        aug_data = torch.reshape(aug_data, (7, 1, 40, 56, 40))

        output = self.module(data)
        output_aug = self.module(aug_data)
        del data, aug_data
        l = trainer.loss(output, target)
        l_aug = trainer.loss(output_aug, aug_target)
        train_losses_epoch.append(l)
        loss_unet = l + 0.5*l_aug

        return loss_unet


class Arch(ImplicitProblem):
    def training_step(self, batch):
        data = batch['data']
        target = batch['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=int(args.gpu_ids))
            target = to_cuda(target, gpu_id=int(args.gpu_ids))        

        data, target[0], target[1], target[2] = \
            data[-2:], target[0][-2:], target[1][-2:], target[2][-2:]


        output = trainer.network(data)
        del data
        loss_arch = trainer.loss(output, target)
        
        return loss_arch


class SSEngine(Engine):

    @torch.no_grad()
    def validation(self):
        trainer.all_tr_losses.append(np.mean(train_losses_epoch))
        train_losses_epoch = []
        trainer.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

        trainer.network.eval()
        val_losses = []
        for b in range(trainer.num_val_batches_per_epoch):
            l = trainer.run_iteration(trainer.val_gen, False, True)
            val_losses.append(l)
        trainer.all_val_losses.append(np.mean(val_losses))
        trainer.print_to_log_file("validation loss: %.4f" % trainer.all_val_losses[-1])

        if trainer.also_val_in_tr_mode:
            trainer.network.train()
            # validation with train=True
            val_losses = []
            for b in range(trainer.num_val_batches_per_epoch):
                l = trainer.run_iteration(trainer.val_gen, False)
                val_losses.append(l)
            trainer.all_val_losses_tr_mode.append(np.mean(val_losses))
            trainer.print_to_log_file("validation loss (train=True): %.4f" % trainer.all_val_losses_tr_mode[-1])

        trainer.update_train_loss_MA()  # needed for lr scheduler and stopping of training
        continue_training = trainer.on_epoch_end()
        trainer.epoch += 1


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=args.unroll_steps)

total_iters = trainer.max_num_epochs * trainer.num_batches_per_epoch
display_freq = trainer.num_batches_per_epoch
engine_config = EngineConfig(
    valid_step=display_freq,
    train_iters=total_iters,
    roll_back=True,
)

netG = Generator(
    name='netG',
    module=trainer.model.netG,
    optimizer=trainer.model.optimizer_G,
    train_data_loader=trainer.tr_gen,
    config=inner_config,
    device=device,
)

netD = Discriminator(
    name='netD',
    module=trainer.model.netD,
    optimizer=trainer.model.optimizer_D,
    train_data_loader=trainer.tr_gen,
    config=inner_config,
    device=device,
)

unet = Unet(
    name='unet',
    module=trainer.network,
    optimizer=trainer.optimizer,
    train_data_loader=trainer.tr_gen,
    config=inner_config,
    device=device,
)

optimizer_arch = torch.optim.Adam(networks.arch_parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-3)
arch = Arch(
    name='arch',
    module=trainer.network,
    optimizer=optimizer_arch,
    train_data_loader=trainer.val_gen,
    config=outer_config,
    device=device,
)

problems = [netG, netD, unet, arch]
l2u = {netG: [unet], unet: [arch]}
u2l = {arch: [netG]}
dependencies = {"l2u": l2u, "u2l": u2l}
engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)


if find_lr:
    trainer.find_lr()
else:
    if not validation_only:
        if args.continue_training:
            # -c was set, continue a previous training and ignore pretrained weights
            trainer.load_latest_checkpoint()
        elif (not args.continue_training) and (args.pretrained_weights is not None):
            # we start a new training. If pretrained_weights are set, use them
            load_pretrained_weights(trainer.network, args.pretrained_weights)
        else:
            # new training without pretraine weights, do nothing
            pass

        if args.train_end2end:
            engine.run()
        trainer.run_training()
        if args.train_end2end:
            engine.run()
        trainer.epoch -= 1
    else:
        if valbest:
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_final_checkpoint(train=False)

    trainer.network.eval()

    if args.disable_validation_inference:
        print("Validation inference was disabled. Not running inference on validation set.")
    else:
        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                        run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                        overwrite=args.val_disable_overwrite)

    if network == '3d_lowres' and not args.disable_next_stage_pred:
        print("predicting segmentations for the next stage of the cascade")
        predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


