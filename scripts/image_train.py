"""
Train a diffusion model on images.
"""

import argparse
import os
import sys
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    create_model_and_diffusion_transformer,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_transformer,
    args_to_dict, add_dict_to_argparser)
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
    if args.use_transformer:
        model, diffusion = create_model_and_diffusion_transformer(
            **args_to_dict(args, model_and_diffusion_defaults_transformer().keys())
        )
    else:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    print('Params number in model: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))))
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
    defaults.update(model_and_diffusion_defaults_transformer())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--gpus", type=int, nargs='+', help="String that contains available GPUs to use", default=[0])
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--use_transformer", default=False, action='store_true')
    return parser


if __name__ == "__main__":
    main()
