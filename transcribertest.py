import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Free up VRAM before loading
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.autograd.set_detect_anomaly(True)

repo_id = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(
    repo_id,
    trust_remote_code=True,
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    repo_id,
    use_safetensors=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    device_map="auto"
)

prompt = "Given the following audio context: <SpeechHere>\n\nText instruction: {query}"
transcribe_query = "Please transcribe this speech."
translate_query = "Can you please translate this speech into written Chinese?"

conversation = [
    [{"role": "user", "content": prompt.format(query=transcribe_query)}],
    [{"role": "user", "content": prompt.format(query=translate_query)}],
]

chat_prompt = processor.tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=False,
    add_generation_prompt=True
)

# Use an audio within 30 seconds, 16000hz.
audio_array, sample_rate = librosa.load(
    "AudioSamples/flight.wav", sr=16000)
audio_array = [audio_array]*2
inputs = processor(text=chat_prompt, audios=audio_array)

for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        inputs[key] = inputs[key].to(device)

        if value.dtype == torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                         temperature=0.1, repetition_penalty=1.1, top_p=0.9, no_repeat_ngram_size=6)
generated_ids = outputs[:, inputs['input_ids'].size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("Transcription Output:", response)
