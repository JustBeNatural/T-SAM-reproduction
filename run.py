import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F
from typing import List
from utils import (load_model,
                       save_image,
                       save_text_sa_avg,
                       get_token_ids,
                       LatentOptConfig)
from models.sd1_5.clip_compat import get_clip_encoder_layers


import argparse
import yaml


def run_one_prompt(
    prompt: str,
    latent_opt_config: LatentOptConfig,
    pipe,
    device: str,
    model_name: str,
    generation_dir: str,
    save_flags: dict,
    seed:int,
    num_inference_steps:int,
    debug_tsam: bool = False,
    negative_prompt: str = None
):
    os.makedirs(generation_dir,exist_ok=True)
    _, eos_idx = get_token_ids(prompt=prompt, tokenizer=pipe.tokenizer)
    for layer in get_clip_encoder_layers(pipe.text_encoder):
        # to get attn score for each prompt
        layer.self_attn.dummy = 0
        

    if save_flags['save_text_selfattn']:
        sa_dir = os.path.join(generation_dir,'text_sa',model_name)
        text_maps = pipe.get_text_sa(prompt=prompt, device=device)
        avg_map = torch.mean(torch.stack([m.detach().clone() for m in text_maps.values()]), dim=0)
        save_text_sa_avg(text_sa = avg_map,
                        directory = sa_dir,
                        file_name = f'{prompt}_block_avg',
                        eos_idx=eos_idx)
        
    
     
    if model_name in ['sd1_5', 'sd1_5x', 'sd1_5x_2']:
        pipe.attn_fetch_x.set_processor(unet = pipe.unet)
    elif model_name in ['pixart', 'pixart_x']:
        pipe.attn_fetch_x.set_processor(transformer = pipe.transformer)
    else:
        raise ValueError("Invalid model name.")
    
    steps_to_save_attention_maps = list(range(num_inference_steps))
    


    max_iter_to_alter = latent_opt_config.max_iter_to_alter
    iterative_refinement_steps = latent_opt_config.iterative_refinement_steps
    if not latent_opt_config.update_latent:
        max_iter_to_alter = 0
        iterative_refinement_steps = []
   
    image, all_maps = pipe(
        prompt=prompt,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=latent_opt_config.guidance_scale,
        negative_prompt=negative_prompt,
        max_iter_to_alter=max_iter_to_alter,
        iterative_refinement_steps=iterative_refinement_steps,
        steps_to_save_attention_maps=steps_to_save_attention_maps,
        latent_opt_config = latent_opt_config,
        debug_tsam=debug_tsam,
    )  
    

    if save_flags['save_crossattn_sim']:
        
        os.makedirs(f"{generation_dir}/{latent_opt_config.cos_sim_dir}",exist_ok=True)
        cosine_similarity_matrix = torch.zeros(eos_idx-1, eos_idx-1)
        map_save_list = [t for t in [0,1,2,3,4,5,6,7,8,9,10,20,30,40,49] if t < num_inference_steps]
        for timestep in map_save_list:
            cross_attn_map = all_maps[0][timestep].reshape(-1, 77)
            for i in range(1,eos_idx):
                for j in range(1,i+1):
                    cosine_sim = F.cosine_similarity(cross_attn_map[:, i], cross_attn_map[:, j], dim=0)
                    cosine_similarity_matrix[i-1, j-1] = cosine_sim
            
            cosine_similarity_matrix_np = cosine_similarity_matrix.numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(cosine_similarity_matrix_np, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title("Cosine Similarity Matrix")
            plt.savefig(f"{generation_dir}/{latent_opt_config.cos_sim_dir}/test_{latent_opt_config.update_latent}_{prompt}_{timestep}_cossim.png")

    if save_flags['save_gen_images']:
        image_save_dir = os.path.join(generation_dir,'images',model_name)
        save_image(image=image[0],directory=image_save_dir, file_name=f'{prompt}_seed_{seed}')


def parse_args():
    parser = argparse.ArgumentParser(description="Run latent optimization")
    parser.add_argument("--prompt", type=str, default='a green glasses and a yellow clock',help="Text prompt")
    parser.add_argument("--model_name", type=str, default="sd1_5x_2", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--steps", type=int, default=50, help="# of inference steps")
    parser.add_argument("--seed", type=int, default=4913, help="Random seed")
    parser.add_argument("--generation_dir", type=str, default="./generation_dir", help="Output dir")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Text prompt for things to avoid")
    parser.add_argument("--no_update_latent", action="store_true", help="Disable T-SAM latent optimization")
    parser.add_argument("--no_save_text_selfattn", action="store_true", help="Do not save text self-attention map")
    parser.add_argument("--no_save_crossattn", action="store_true", help="Do not save cross-attention similarity maps")
    parser.add_argument("--no_save_images", action="store_true", help="Do not save generated images")
    parser.add_argument("--max_iter_to_alter", type=int, default=None, help="Override latent optimization step count")
    parser.add_argument("--no_iterative_refinement", action="store_true", help="Disable extra iterative refinement steps")
    parser.add_argument("--debug_tsam", action="store_true", help="Print whether T-SAM loss and latent updates are active")
    parser.add_argument("--debug_prompt", action="store_true", help="Print the exact prompt and output directory")
    return parser.parse_args()



if __name__ == "__main__":
    with open('configs/config.yaml','r') as f:
        config_dict = yaml.safe_load(f)
        
    latent_opt_config = LatentOptConfig(**config_dict)
    args = parse_args()
    if args.no_update_latent:
        latent_opt_config.update_latent = False
        latent_opt_config.max_iter_to_alter = 0
        latent_opt_config.iterative_refinement_steps = []
    if args.max_iter_to_alter is not None:
        latent_opt_config.max_iter_to_alter = args.max_iter_to_alter
    if args.no_iterative_refinement:
        latent_opt_config.iterative_refinement_steps = []
    prompt_list = [args.prompt]
    if args.debug_prompt:
        print(f"Prompt received: {args.prompt!r}")
        print(f"Negative prompt received: {args.negative_prompt!r}")
        print(f"Output directory: {args.generation_dir!r}")
    save_flags = {
        "save_text_selfattn": not args.no_save_text_selfattn,
        "save_gen_images": not args.no_save_images,
        "save_crossattn_sim": not args.no_save_crossattn
    }
    
    pipe = load_model(model_name=args.model_name, device=args.device)
    for idx, prompt in enumerate(prompt_list):
        run_one_prompt(
            prompt=prompt,
            pipe=pipe,
            latent_opt_config=latent_opt_config,
            device=args.device,
            generation_dir=args.generation_dir,
            seed=args.seed,
            num_inference_steps=args.steps,
            save_flags=save_flags,
            model_name=args.model_name,
            debug_tsam=args.debug_tsam,
            negative_prompt=args.negative_prompt
        )
