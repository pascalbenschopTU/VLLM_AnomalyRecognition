## Folder Setup
In this folder you need to place the weights of the models.

- Gemma3-4B/
- Qwen2.5-VL-7B-Instruct
- VideoLLama3-7B

These can be downloaded with the `download_hf_model.py` script

And NVILA-8B-Video, which has to be placed in the VILA/ folder (which you can clone from https://github.com/NVlabs/VILA)


## Evaluation scripts
For Gemma3, Qwen2.5 and VideoLLama3 the `apply_videollm.py` script evaluates models on videos by using different prompts. See the jobs for different configurations.
For NVILA the `apply_vila.py` script should be placed inside the VILA folder since it requires certain dependencies.
