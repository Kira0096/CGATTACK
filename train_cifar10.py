import argparse
import torch
# import Learner
import datasets
import utils
import ast

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='train c-Glow')

    # input output path
    parser.add_argument("-d", "--dataset_name", type=str, default="horse")
    parser.add_argument("-r", "--dataset_root", type=str, default="")

    # log root
    parser.add_argument("--log_root", type=str, default="")

    # C-Glow parameters
    parser.add_argument("--x_size", type=tuple, default=(3,32,32))
    parser.add_argument("--y_size", type=tuple, default=(3,32,32))
    parser.add_argument("--x_hidden_channels", type=int, default=128)
    parser.add_argument("--x_hidden_size", type=int, default=64)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=ast.literal_eval, default=False)


    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.0)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=2.0)


    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--betas", type=tuple, default=(0.9,0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--regularizer", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--Lambda", type=float, default=1e-2)

    # Trainer parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--max_grad_clip", type=float, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--checkpoints_gap", type=int, default=1000)
    parser.add_argument("--nll_gap", type=int, default=1)
    parser.add_argument("--inference_gap", type=int, default=1000)
    parser.add_argument("--save_gap", type=int, default=1000)
    parser.add_argument("--adv_loss", type=ast.literal_eval, default=False)
    parser.add_argument("--target", type=ast.literal_eval, default=False)
    parser.add_argument("--tanh", type=ast.literal_eval, default=False)
    parser.add_argument("--only", type=ast.literal_eval, default=False)
    parser.add_argument("--clamp", type=ast.literal_eval, default=False)
    parser.add_argument("--ref", type=str, default="pyramidnet")
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--class_size", type=int, default=-1)
    parser.add_argument("--label", type=int, default=0)

    # Adv augmentation
    parser.add_argument("--adv_aug", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_rand", type=ast.literal_eval, default=False)
    parser.add_argument("--nes", type=ast.literal_eval, default=False)
    parser.add_argument("--new_form", type=ast.literal_eval, default=False)
    parser.add_argument("--normalize_grad", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_epoch", type=int, default=0)

    # model path
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--name", type=str, default="")


    args = parser.parse_args()
    cuda = torch.cuda.is_available()

    training_set = datasets.cifar10(args.dataset_root, (args.y_size[1], args.y_size[2]), args.y_size[0], "train")
    valid_set = datasets.cifar10(args.dataset_root, (args.y_size[1], args.y_size[2]), args.y_size[0], portion="valid")


    from CGlowModel import CondGlowModel

    model = CondGlowModel(args)
    if cuda:
        model = model.cuda()

    from utils import count_parameters
    print("number of param: {}".format(count_parameters(model)))

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,betas=args.betas, weight_decay=args.regularizer)

    # scheduler
    scheduler = None

    if args.model_path != "":
        state = utils.load_state(args.model_path, cuda)
        optim.load_state_dict(state["optim"])
        model.load_state_dict(state["model"])
        args.steps = state["iteration"] + 1
        if scheduler is not None and state.get("scheduler", None) is not None:
            scheduler.load_state_dict(state["scheduler"])
        del state
        for name, param in model.named_parameters():
            if 'class_Con' in name:
                param.requires_grad = False
        

    # begin to train
    if args.adv_loss:
        adv_model = utils.load_adv(args.ref.split(","))
        import RLearner
        trainer = RLearner.Trainer(model, adv_model, optim, scheduler, training_set, valid_set, args, cuda)
    else:
        import Learner
        trainer = Learner.Trainer(model, optim, scheduler, training_set, valid_set, args, cuda)

    
    trainer.train()
