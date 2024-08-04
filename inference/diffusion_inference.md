# Evaluation Strategy 

### Background & Goals
We have identified failure modes in SDXL, SD3M, and PixArt Sigma. Building on this (via a qualitative analysis), we will understand how our diffusion model mitigates some of these failures (and present persistent issues if any.) A subset of high-quality prompts will serve as an internal benchmark of sorts for rapid testing.  

Image generation is usually evaluated for image quality and image-text alignment. With this in mind, we will set up specific evaluations while concurrently fine-tuning the diffusion model. This will help us to understand what to look for while conducting ablation studies on model design parameters.

For starters, we plan to use the GenEval as our initial benchmark.
(Additional evals for safe content and bias/stereotypes might be required.)

### Failure Modes
* Counts 
* Shape attribution
* Color attribution
* Text/Text Rendering/Typography
* Object relations (starting with simple 2-object cases)
* Human anatomy
* Human pose/ actions



### Qualitative Analysis (TBD)
A set of high-quality prompts that we can use for quick testing of our T2I diffusion (via Kevin)


### Quantitative Evals (TBD)
* FID (image quality): Fidelity, diversity ++.
* CLIP(image-text alignment): Alignment core via vision-language pretrained models.
* TIFA : Downstream evaluation using another model (VQA) on the images.
* Object-centric metrics [Hinz et al., 2020, Cho et al., 2022] leveraging object detectors such as DETR [Carion et al., 2020].
