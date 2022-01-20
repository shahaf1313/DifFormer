"""
Train a diffusion model on images.
"""

import argparse
import os
import sys
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict, add_dict_to_argparser
from guided_diffusion.train_util import TrainLoop
import datetime

def main():
    args = create_argparser().parse_args()
    command_line = ''
    for s in sys.argv:
        command_line += s + ' '
    args.command_line = command_line
    dist_util.setup_dist(args.gpus)
    log_root_dir = '/home/shahaf/guided_diffusion/%sruns' % ('debug_' if args.debug else '')
    logger.configure(dir=os.path.join(log_root_dir,datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-GPU' + str(args.gpus[0])))
    logger.log('configuration of current run:')
    for argument in vars(args):
        logger.log(argument + ': ' + str(getattr(args, argument)))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    logger.log(model)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

# def diffusion_defaults():
#     """
#     Defaults for image and classifier training.
#     """
#     return dict(
#         learn_sigma=False,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=False,
#         rescale_learned_sigmas=False,
#     )
#
#
# def classifier_defaults():
#     """
#     Defaults for classifier models.
#     """
#     return dict(
#         image_size=64,
#         classifier_use_fp16=False,
#         classifier_width=128,
#         classifier_depth=2,
#         classifier_attention_resolutions="32,16,8",  # 16
#         classifier_use_scale_shift_norm=True,  # False
#         classifier_resblock_updown=True,  # False
#         classifier_pool="attention",
#     )

#
# def classifier_and_diffusion_defaults():
#     res = classifier_defaults()
#     res.update(diffusion_defaults())
#     return res
#
#
#
# def add_dict_to_argparser(parser, default_dict):
#     for k, v in default_dict.items():
#         v_type = type(v)
#         if v is None:
#             v_type = str
#         elif isinstance(v, bool):
#             v_type = str2bool
#         parser.add_argument(f"--{k}", default=v, type=v_type)
#
#
# def args_to_dict(args, keys):
#     return {k: getattr(args, k) for k in keys}
#
#
# def str2bool(v):
#     """
#     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
#     """
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("boolean value expected")

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--gpus", type=int, nargs='+', help="String that contains available GPUs to use", default=[0])
    parser.add_argument("--debug", default=False, action='store_true')
    return parser


if __name__ == "__main__":
    main()
