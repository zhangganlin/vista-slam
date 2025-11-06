from vista_slam.sta_model.train import get_args_parser, train

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    train(args)