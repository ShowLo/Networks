def get_model(args):
    if args.mode == 'normal':
        from .Normal.get_model import get_model
    elif args.mode == 'ac':
        from .AC.get_model import get_model
    elif args.mode == 'es':
        from .ES.get_model import get_model
    else:
        raise ValueError('mode: ' + args.mode + ' is wrong!')
    return get_model(args)