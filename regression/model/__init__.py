from model.cavia_rl import CaviaRL


def get_model_type(model_type, is_rl=False):
    if model_type == "CAVIA":
        MODEL = CaviaRL if is_rl else Cavia
    else:
        raise ValueError()

    return MODEL
