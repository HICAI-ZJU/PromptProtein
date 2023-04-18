import torch


def openprotein_promptprotein(checkpoint_dir):
    from collections import OrderedDict
    from .dictionary_promptprotein import Alphabet
    from .promptprotein import PromptProtein

    dictionary = Alphabet.build_alphabet()
    model_data = torch.load(f"{checkpoint_dir}", map_location="cpu")

    model_cfg = model_data['cfg']['model']
    model_state = model_data['model']
    model_type = PromptProtein

    model = model_type(
        model_cfg,
        dictionary
    )

    with torch.no_grad():
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            if 'encoder' in k:
                k = k.replace('encoder.', '')                    
                new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    return model, dictionary
