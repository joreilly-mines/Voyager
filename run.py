from voyager import Voyager

import torch

torch.cuda.empty_cache()

mc_port = 25565

openai_api_key = "0"

voyager = Voyager(
    mc_port=mc_port,
    openai_api_key=openai_api_key
)

voyager.learn()