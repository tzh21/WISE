import torch
from diffusers import Flux2KleinPipeline

device = "cuda:0"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("/share/project/tzh/models/FLUX.2-klein-4B", torch_dtype=dtype)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]
image.save("flux-klein.png")
