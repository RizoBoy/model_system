from model import Net
import torch
import torchaudio
import time
import numpy as np
import json
import os
from utils import glob_audio_files
from tqdm import tqdm


def load_model(checkpoint_path, config_path):
    with open(config_path) as f:
        config = json.load(f)
    model = Net(**config['model_params'])
    model.load_state_dict(torch.load(
        checkpoint_path, map_location="cpu")['model'])
    return model, config['data']['sr']


def load_audio(audio_path, sample_rate):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(0, keepdim=False)
    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio


def save_audio(audio, audio_path, sample_rate):
    torchaudio.save(audio_path, audio, sample_rate)


def infer(model, audio):
    return model(audio.unsqueeze(0).unsqueeze(0)).squeeze(0)


def infer_stream(model, audio, chunk_factor, sr):
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor
    # pad audio to be a multiple of L * dec_chunk_size
    original_len = len(audio)
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    # scoot audio down by L
    audio = torch.cat((audio[L:], torch.zeros(L)))
    audio_chunks = torch.split(audio, chunk_len)
    # add lookahead context from prev chunk
    new_audio_chunks = []
    for i, a in enumerate(audio_chunks):
        if i == 0:
            front_ctx = torch.zeros(L * 2)
        else:
            front_ctx = audio_chunks[i - 1][-L * 2:]
        new_audio_chunks.append(torch.cat([front_ctx, a]))
    audio_chunks = new_audio_chunks

    outputs = []
    times = []
    with torch.inference_mode():
        enc_buf, dec_buf, out_buf = model.init_buffers(
            1, torch.device('cpu'))
        if hasattr(model, 'convnet_pre'):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(
                1, torch.device('cpu'))
        else:
            convnet_pre_ctx = None
        for chunk in audio_chunks:
            start = time.time()
            output, \
                enc_buf, dec_buf, out_buf, \
                convnet_pre_ctx = model(chunk.unsqueeze(
                    0).unsqueeze(0),
                    enc_buf, dec_buf, out_buf,
                    convnet_pre_ctx, pad=(not model.lookahead)
                )
            outputs.append(output)
            times.append(time.time() - start)
        # concatenate outputs
    outputs = torch.cat(outputs, dim=2)
    # Calculate RTF
    avg_time = np.mean(times)
    rtf = (chunk_len / sr) / avg_time
    # calculate e2e latency
    e2e_latency = ((2 * L + chunk_len) / sr + avg_time) * 1000
    # remove padding
    outputs = outputs[:, :, :original_len].squeeze(0)
    return outputs, rtf, e2e_latency


def do_infer(model, audio, chunk_factor, sr, stream):
    with torch.no_grad():
        if stream:
            outputs, rtf, e2e_latency = infer_stream(
                model, audio, chunk_factor, sr)
            return outputs, rtf, e2e_latency
        else:
            outputs = infer(model, audio)
            rtf = None
            e2e_latency = None
    return outputs, rtf, e2e_latency


def main():
    dir_test ='test_sample'
    model, sr = load_model("llvc_models/models/checkpoints/llvc/G_500000.pth", 'experiments/llvc/config.json')
    # os.mkdir('test_sample')
    # check if fname is a directory
    audio = load_audio('test_wavs/174-50561-0000.wav', sr)
    out, rtf, e2e_latency = do_infer(
        model, audio, 1, sr, True
    )
    # out_fname = os.path.join(
        # args.out_dir, os.path.basename(args.fname))
    save_audio(out, f'{dir_test}/test.wav',sr)


if __name__ == '__main__':
    main()
