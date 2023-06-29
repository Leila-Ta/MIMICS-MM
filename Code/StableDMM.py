import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "CompVis/stable-diffusion-v1-4"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

StableD_file = open("./MMCQStableDremaining.txt")
read_Score_file = csv.reader(StableD_file, delimiter="\t")

for i in read_Score_file:
    image = pipe(i).images[0]
    image.save(str(i) + ".png")