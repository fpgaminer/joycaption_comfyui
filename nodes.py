import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import folder_paths
import comfy.model_management as model_management
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage


# From (https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/1ca496c1c8e8ada94d7d2644b8a7d4b3dc9729b3/nodes/qwen2vl.py)
# Apache 2.0 License
MEMORY_EFFICIENT_CONFIGS = {
	"Default": {},
	"Balanced (8-bit)": {
		"load_in_8bit": True,
	},
	"Maximum Savings (4-bit)": {
		"load_in_4bit": True,
		"bnb_4bit_quant_type": "nf4",
		"bnb_4bit_compute_dtype": torch.bfloat16,
		"bnb_4bit_use_double_quant": True,
	},
}


CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a detailed description for this image.",
		"Write a detailed description for this image in {word_count} words or less.",
		"Write a {length} detailed description for this image.",
	],
	"Descriptive (Casual)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Straightforward": [
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
	],
	"Stable Diffusion Prompt": [
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
		"Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Danbooru tag list": [
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
	],
	"e621 tag list": [
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
	],
	"Rule34 tag list": [
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}

EXTRA_OPTIONS = [
	"",
	"If there is a person/character in the image you must refer to them as {name}.",
	"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
	"Include information about lighting.",
	"Include information about camera angle.",
	"Include information about whether there is a watermark or not.",
	"Include information about whether there are JPEG artifacts or not.",
	"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
	"Do NOT include anything sexual; keep it PG.",
	"Do NOT mention the image's resolution.",
	"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
	"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
	"Do NOT mention any text that is in the image.",
	"Specify the depth of field and whether the background is in focus or blurred.",
	"If applicable, mention the likely use of artificial or natural lighting sources.",
	"Do NOT use any ambiguous language.",
	"Include whether the image is sfw, suggestive, or nsfw.",
	"ONLY describe the most important elements of the image.",
	"If it is a work of art, do not include the artist's name or the title of the work.",
	"Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
	"""Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
	"Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
	"Include information about the ages of any people/characters when applicable.",
	"Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
	"Do not mention the mood/feeling/etc of the image.",
	"Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
	"If there is a watermark, you must mention it.",
	"""Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.""",
]

CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"] + [
	str(i) for i in range(20, 261, 10)
]


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
	# Choose the right template row in CAPTION_TYPE_MAP
	if caption_length == "any":
		map_idx = 0
	elif isinstance(caption_length, str) and caption_length.isdigit():
		map_idx = 1  # numeric-word-count template
	else:
		map_idx = 2  # length descriptor template

	prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

	if extra_options:
		prompt += " " + " ".join(extra_options)

	return prompt.format(
		name=name_input or "{NAME}",
		length=caption_length,
		word_count=caption_length,
	)


class JoyCaptionPredictor:
	def __init__(self, model: str, memory_mode: str):
		checkpoint_path = Path(folder_paths.models_dir) / "LLavacheckpoints" / Path(model).stem
		if not checkpoint_path.exists():
			# Download the model
			from huggingface_hub import snapshot_download

			snapshot_download(
				repo_id=model, local_dir=str(checkpoint_path), force_download=False, local_files_only=False
			)

		self.checkpoint_path = str(checkpoint_path)
		self.memory_mode = memory_mode

		self.inference_device = model_management.get_torch_device()
		self.offload_device = model_management.unet_offload_device()

		self.processor = AutoProcessor.from_pretrained(str(checkpoint_path))

		self.model = None
		self.model_size_bytes = None
		self.is_kbit = self.memory_mode != "Default"

	def _load_model(self):
		# In normal mode:
		#     We load the model, free memory on the offload device, and then move it to the offload device.
		# In quantized modes:
		#     The model must be loaded directory to the inference device.
		#     This function is only called during inference.
		#     After inference, if we need to offload, we just unload the model entirely.
		#     It'll be rebuilt during the next inference.
		#     We free memory on the inference device if we know how big the model is from a previous load.
		if self.memory_mode == "Default":
			self.model = LlavaForConditionalGeneration.from_pretrained(self.checkpoint_path, torch_dtype="bfloat16")
			self.model_size_bytes = model_management.module_size(self.model)
			model_management.free_memory(self.model_size_bytes, self.offload_device)
			self.model.to(self.offload_device)
		else:
			from transformers import BitsAndBytesConfig

			if self.model_size_bytes is not None:
				model_management.free_memory(self.model_size_bytes, self.inference_device)

			qnt_config = BitsAndBytesConfig(
				**MEMORY_EFFICIENT_CONFIGS[self.memory_mode],
				llm_int8_skip_modules=[
					"vision_tower",
					"multi_modal_projector",
				],  # Transformer's Siglip implementation has bugs when quantized, so skip those.
			)

			self.model = LlavaForConditionalGeneration.from_pretrained(
				self.checkpoint_path,
				torch_dtype="auto",
				quantization_config=qnt_config,
				device_map=_cuda_device_map(self.inference_device),
			)
			self.model_size_bytes = model_management.module_size(self.model)

		self.model.eval()

		print(f"Loaded model (mode={self.memory_mode}, kbit={self.is_kbit})")

	def prepare_for_inference(self):
		if self.model is None:
			self._load_model()

		if self.is_kbit:
			return

		model_management.free_memory(self.model_size_bytes, self.inference_device)
		self.model.to(self.inference_device)

	def cleanup_after_inference(self, keep_loaded: bool):
		if keep_loaded:
			return
		if self.model is None:
			return

		if self.is_kbit:
			self.unload()
			return

		self.model.to(self.offload_device)
		model_management.soft_empty_cache()

	def unload(self):
		if self.model is not None:
			del self.model
			self.model = None
		model_management.soft_empty_cache()

	@torch.inference_mode()
	def generate(
		self,
		image: Image.Image,
		system: str,
		prompt: str,
		max_new_tokens: int,
		temperature: float,
		top_p: float,
		top_k: int,
	) -> str:
		# Load the model if it isn't already loaded and move it to the inference device if needed.
		self.prepare_for_inference()

		convo = [
			{
				"role": "system",
				"content": system.strip(),
			},
			{
				"role": "user",
				"content": prompt.strip(),
			},
		]

		# Format the conversation
		convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
		assert isinstance(convo_string, str)

		# Keep processor tensors on the same device as the loaded model.
		inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.inference_device)
		model_dtype = getattr(self.model, "dtype", None)
		if (
			"pixel_values" in inputs
			and isinstance(model_dtype, torch.dtype)
			and torch.is_floating_point(inputs["pixel_values"])
		):
			inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

		# Generate the captions
		device_type = model_management.get_autocast_device(self.inference_device)
		autocast_available = torch.amp.autocast_mode.is_autocast_available(device_type)
		bf16_supported = (device_type != "cuda") or torch.cuda.is_bf16_supported()

		with torch.autocast(
			device_type=device_type, dtype=torch.bfloat16, enabled=autocast_available and bf16_supported
		):
			generate_ids = self.model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				do_sample=True if temperature > 0 else False,
				suppress_tokens=None,
				use_cache=True,
				temperature=temperature,
				top_k=None if top_k == 0 else top_k,
				top_p=top_p,
			)[0]

		# Trim off the prompt
		generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

		# Decode the caption
		caption = self.processor.tokenizer.decode(
			generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)
		return caption.strip()


class JoyCaption:
	@classmethod
	def INPUT_TYPES(cls):
		# fmt: off
		req = {
			"image":          ("IMAGE",),
			"memory_mode":    (list(MEMORY_EFFICIENT_CONFIGS.keys()),),
			"caption_type":   (list(CAPTION_TYPE_MAP.keys()),),
			"caption_length": (CAPTION_LENGTH_CHOICES,),

			"extra_option1":  (list(EXTRA_OPTIONS),),
			"extra_option2":  (list(EXTRA_OPTIONS),),
			"extra_option3":  (list(EXTRA_OPTIONS),),
			"extra_option4":  (list(EXTRA_OPTIONS),),
			"extra_option5":  (list(EXTRA_OPTIONS),),
			"person_name":    ("STRING", {"default": "", "multiline": False, "placeholder": "only needed if you use the 'If there is a person/character in the image you must refer to them as {name}.' extra option."}),

			# generation params
			"max_new_tokens": ("INT",     {"default": 512, "min": 1,   "max": 2048}),
			"temperature":    ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
			"top_p":          ("FLOAT",   {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
			"top_k":          ("INT",     {"default": 0,   "min": 0,   "max": 100}),
			"keep_loaded":    ("BOOLEAN", {"default": False}),
		}
		# fmt: on

		return {"required": req}

	RETURN_TYPES = ("STRING", "STRING")
	RETURN_NAMES = ("query", "caption")
	FUNCTION = "generate"
	CATEGORY = "JoyCaption"

	def __init__(self):
		self.predictor = None

	def generate(
		self,
		image,
		memory_mode,
		caption_type,
		caption_length,
		extra_option1,
		extra_option2,
		extra_option3,
		extra_option4,
		extra_option5,
		person_name,
		max_new_tokens,
		temperature: float,
		top_p: float,
		top_k: int,
		keep_loaded: bool,
	):
		if image.shape[0] != 1:
			return ("", "Error: batch size greater than 1 is not supported.")

		# load / swap the model if needed
		if self.predictor is None or self.predictor.memory_mode != memory_mode:
			if self.predictor is not None:
				self.predictor.unload()
				del self.predictor
				self.predictor = None

			try:
				self.predictor = JoyCaptionPredictor("fancyfeast/llama-joycaption-beta-one-hf-llava", memory_mode)
			except Exception as e:
				return ("", f"Error loading model: {e}")

		extras = [extra_option1, extra_option2, extra_option3, extra_option4, extra_option5]
		extras = [extra for extra in extras if extra]
		prompt = build_prompt(caption_type, caption_length, extras, person_name)
		system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."

		# This is a bit silly. We get the image as a tensor, and we could just use that directly (just need to resize and adjust the normalization).
		# But JoyCaption was trained on images that were resized using lanczos, which I think PyTorch doesn't support.
		# Just to be safe, we'll convert the image to a PIL image and let the processor handle it correctly.
		pil_image = ToPILImage()(image[0].permute(2, 0, 1))
		try:
			response = self.predictor.generate(
				image=pil_image,
				system=system_prompt,
				prompt=prompt,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
			)
		finally:
			self.predictor.cleanup_after_inference(keep_loaded=keep_loaded)

		return (prompt, response)


class JoyCaptionCustom:
	@classmethod
	def INPUT_TYPES(cls):
		# fmt: off
		return {
			"required": {
				"image":          ("IMAGE",),
				"memory_mode":    (list(MEMORY_EFFICIENT_CONFIGS.keys()),),
				"system_prompt":  ("STRING", {"multiline": False, "default": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions." }),
				"user_query":     ("STRING", {"multiline": True, "default": "Write a detailed description for this image." }),
				# generation params
				"max_new_tokens": ("INT",     {"default": 512, "min": 1,   "max": 2048}),
				"temperature":    ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
				"top_p":          ("FLOAT",   {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
				"top_k":          ("INT",     {"default": 0,   "min": 0,   "max": 100}),
				"keep_loaded":    ("BOOLEAN", {"default": False}),
			},
		}
		# fmt: on

	RETURN_TYPES = ("STRING",)
	FUNCTION = "generate"
	CATEGORY = "JoyCaption"

	def __init__(self):
		self.predictor = None

	def generate(
		self,
		image,
		memory_mode,
		system_prompt: str,
		user_query: str,
		max_new_tokens: int,
		temperature: float,
		top_p: float,
		top_k: int,
		keep_loaded: bool,
	):
		if image.shape[0] != 1:
			return ("Error: batch size greater than 1 is not supported.",)

		if self.predictor is None or self.predictor.memory_mode != memory_mode:
			if self.predictor is not None:
				self.predictor.unload()
				del self.predictor
				self.predictor = None

			try:
				self.predictor = JoyCaptionPredictor("fancyfeast/llama-joycaption-beta-one-hf-llava", memory_mode)
			except Exception as e:
				return (f"Error loading model: {e}",)

		# This is a bit silly. We get the image as a tensor, and we could just use that directly (just need to resize and adjust the normalization).
		# But JoyCaption was trained on images that were resized using lanczos, which I think PyTorch doesn't support.
		# Just to be safe, we'll convert the image to a PIL image and let the processor handle it correctly.
		pil_image = ToPILImage()(image[0].permute(2, 0, 1))
		try:
			response = self.predictor.generate(
				image=pil_image,
				system=system_prompt,
				prompt=user_query,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
			)
		finally:
			self.predictor.cleanup_after_inference(keep_loaded=keep_loaded)

		return (response,)


def _cuda_device_map(dev: torch.device):
	if dev.type == "cuda":
		return {"": (dev.index or 0)}
	return {"": str(dev)}
