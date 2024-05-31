from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.conf import settings
from django.utils.datastructures import MultiValueDictKeyError
from django.templatetags.static import static

import numpy as np
import cv2
import keras
import pypianoroll
import torch
from midi2audio import FluidSynth

from keras.preprocessing.image import load_img
from tensorflow.python.keras.backend import set_session

from .models import ConvVAE, MelodyNN, ConditionalNN, ExternalSongGenerationDataset

def index(request):
    if request.method == 'POST':
        try:
            file = request.FILES['imageFile']
            file_name = default_storage.save(file.name, file)
            file_path = default_storage.path(file_name)

            image = load_img(file_path)

            faces = settings.FACE_CASCADER.detectMultiScale(cv2.imread(file_path), 1.3, 5)
            print("DEBUG: ", image.size)

            for (x, y, w, h) in faces: 
                image = image.crop((x, y, x + w, y + h))
                image = image.resize((224, 224))
                break
            
            if len(faces) == 0:
                image = image.resize((224, 224))

            # Save the image to see if it is cropped correctly
            #image.save('test.jpg', 'JPEG')

            numpy_image = np.array(image)
            numpy_image = np.array(numpy_image).astype('float32') / 255
            image_batch = np.expand_dims(numpy_image, axis=0)

            predictions = settings.IMAGE_MODEL.predict(image_batch)
            predictedMood = decode_emotions(predictions)

            print("[DEBUG] Predicted: ", predictedMood)

            generate_song(predictions, file_name)

            audioFilePath = f'generated/{file_name}.wav'
            print("[DEBUG] Audio file path: ", audioFilePath)

            return render(request, 'index.html', {'predictions': predictedMood, 'audioFilePath': audioFilePath, 'labeledEmotion': decode_emotions(predictions)})
        except MultiValueDictKeyError as e:
            print("[ERROR] no file seleced")
            return render(request, 'index.html')
    else:
        return render(request, 'index.html')
        
    return render(request, 'index.html')

def decode_emotions(predictions):
    emotions = ['angry', 'angry', 'scary', 'happy', 'neutral', 'sad', 'surprising']
    return emotions[np.argmax(predictions)]

# Operative part ;)

def generate_song(mood_vector, song_name):
    selected_song = song_selection(mood_vector)
    pianorolls = build_pianoroll(selected_song)

    vae_models, nn_models = load_vaes()
    
    sample = pianorolls[:, :32, :]

    song_dataset = ExternalSongGenerationDataset(pianorolls, seq_length = 32)
    song_loader = torch.utils.data.DataLoader(song_dataset, batch_size = 1, shuffle = False)
    song_latent_list = generate_latent_samples(song_loader, vae_models)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prediction_steps = 8
    generated_track = torch.zeros((5, 32 * (prediction_steps+1), 128)).to(device)

    for i in range(1, prediction_steps+1):
        sample = generate_music_vae(sample, vae_models, nn_models, noise_sd = 1, threshold = 0.2, binarize = True, latent_samples = song_latent_list, latent_sample_factor = 0.5)
        generated_track[:, (32*i):(32*(i+1)), :] = sample

    generated_track_out = generated_track * 127

    piano_track = pypianoroll.StandardTrack(name = 'Piano', program = 0, is_drum = False, pianoroll = generated_track_out[0, :, :].detach().cpu().numpy())
    guitar_track = pypianoroll.StandardTrack(name = 'Guitar', program = 24, is_drum = False, pianoroll = generated_track_out[1, :, :].detach().cpu().numpy())
    bass_track = pypianoroll.StandardTrack(name = 'Bass', program = 32, is_drum = False, pianoroll = generated_track_out[2, :, :].cpu().detach().numpy())
    strings_track = pypianoroll.StandardTrack(name = 'Strings', program = 48, is_drum = False, pianoroll = generated_track_out[3, :, :].cpu().detach().numpy())
    drums_track = pypianoroll.StandardTrack(name = 'Drums', is_drum = True, pianoroll = generated_track_out[4, :, :].cpu().detach().numpy())
    generated_multitrack = pypianoroll.Multitrack(name = 'Generated', resolution = 2, tracks = [piano_track, guitar_track, bass_track, strings_track, drums_track])

    generated_pm = pypianoroll.to_pretty_midi(generated_multitrack)
    generated_pm.write(f'media/generated/{song_name}.mid')
    
    fs = FluidSynth()
    fs.midi_to_audio(f'media/generated/{song_name}.mid', f'media/generated/{song_name}.wav')

def song_selection(mood_vector):
    mood_class = np.argmax(mood_vector)
    
    map_class_mood = {0: 'AF',  1 :'AN',2: 'DI',3: 'HA',4: 'NE',5: 'SA',6: 'SU'}
    pools = settings.POOL_DICT

    chosen_emotion = map_class_mood[mood_class]
    chosen_pool = pools[chosen_emotion]

    chosen_id = np.random.randint(0, len(chosen_pool))
    chosen_song = chosen_pool[chosen_id]

    return chosen_song

def build_pianoroll(song_name):
    combined_pianorolls = []

    multitrack = pypianoroll.load("datas/" + song_name + '.npz')
    multitrack.set_resolution(2).pad_to_same()

    parts = {'piano_part': None, 'guitar_part': None, 'bass_part': None, 'strings_part': None, 'drums_part': None}
    song_length = None
    empty_array = None
    has_empty_parts = False
    for track in multitrack.tracks:
        if track.name == 'Drums':
            parts['drums_part'] = track.pianoroll
        if track.name == 'Piano':
            parts['piano_part'] = track.pianoroll
        if track.name == 'Guitar':
            parts['guitar_part'] = track.pianoroll
        if track.name == 'Bass':
            parts['bass_part'] = track.pianoroll
        if track.name == 'Strings':
            parts['strings_part'] = track.pianoroll
        if track.pianoroll.shape[0] > 0:
            empty_array = np.zeros_like(track.pianoroll)

    for k,v in parts.items():
        if v.shape[0] == 0:
            parts[k] = empty_array.copy()
        has_empty_parts = True

    # Stack all together - Piano, Guitar, Bass, Strings, Drums
    combined_pianoroll = torch.tensor([parts['piano_part'], parts['guitar_part'], parts['bass_part'], parts['strings_part'], parts['drums_part']])
    combined_pianorolls.append(combined_pianoroll)

    print(f"[DEBUG] Combined pianorolls: {len(combined_pianorolls)}")

    pianoroll_lengths = [e.size()[1] for e in combined_pianorolls]
    combined_pianorolls = torch.hstack(combined_pianorolls)

    torch.save(combined_pianorolls, 'tmp/conditional_pianorolls.pt')
    pianoroll_lengths = torch.tensor(pianoroll_lengths)
    torch.save(pianoroll_lengths, 'tmp/conditional_pianorolls_lengths.pt')

    # Loading the entire dataset
    combined_pianorolls = torch.load('tmp/conditional_pianorolls.pt')
    pianoroll_lengths = torch.load('tmp/conditional_pianorolls_lengths.pt')
    pianoroll_lengths = pianoroll_lengths.numpy()
    pianoroll_cum_lengths = pianoroll_lengths.cumsum()

    # Normalize
    combined_pianorolls = combined_pianorolls / 127.0
    print(combined_pianorolls.shape)

    # Remake the list of pianorolls - ensuring all songs are multiple of 32
    pianorolls_list = []
    pianorolls_list.append(combined_pianorolls[:, :(pianoroll_cum_lengths[0] - pianoroll_cum_lengths[0] % 32), :])
    for i in range(len(pianoroll_cum_lengths) - 1):
        length = pianoroll_cum_lengths[i+1] - pianoroll_cum_lengths[i]
        # Get the nearest multiple of 32
        length_multiple = length - (length % 32)
        pianoroll = combined_pianorolls[:, pianoroll_cum_lengths[i]:(pianoroll_cum_lengths[i] + length_multiple), :]
        pianorolls_list.append(pianoroll)

    # Combine the pianorolls again
    combined_pianorolls = torch.hstack(pianorolls_list)

    return combined_pianorolls

import random

def generate_music_vae(sample, vae_models, nn_models, noise_sd = 0, threshold = 0.3, binarize = True, latent_samples = None, latent_sample_factor = 0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    piano_vae, guitar_vae, bass_vae, strings_vae, drums_vae = vae_models
    melody_nn, guitar_nn, bass_nn, strings_nn, drums_nn = nn_models

    piano, guitar, bass, strings, drums = sample[0, :, :], sample[1, :, :], sample[2, :, :], sample[3, :, :], sample[4, :, :]

    # Convert all part from image space to latent space - {instr}_latent: batch_size x K
    piano_latent = piano_vae.infer(piano.unsqueeze(0).to(device))[:, :-1]
    guitar_latent = guitar_vae.infer(guitar.unsqueeze(0).to(device))[:, :-1]
    bass_latent = bass_vae.infer(bass.unsqueeze(0).to(device))[:, :-1]
    strings_latent = strings_vae.infer(strings.unsqueeze(0).to(device))[:, :-1]
    drums_latent = drums_vae.infer(drums.unsqueeze(0).to(device))[:, :-1]

    if latent_samples: # Choose a random
        piano_latent_sample = random.choice(latent_samples['piano'])
        guitar_latent_sample = random.choice(latent_samples['guitar'])
        bass_latent_sample = random.choice(latent_samples['bass'])
        strings_latent_sample = random.choice(latent_samples['strings'])
        drums_latent_sample = random.choice(latent_samples['drums'])

    # Interpolate between the past latent and sample latent
    piano_latent = latent_sample_factor * piano_latent_sample + (1-latent_sample_factor) * piano_latent
    piano_latent = latent_sample_factor * guitar_latent_sample + (1-latent_sample_factor) * guitar_latent
    piano_latent = latent_sample_factor * bass_latent_sample + (1-latent_sample_factor) * bass_latent
    piano_latent = latent_sample_factor * strings_latent_sample + (1-latent_sample_factor) * strings_latent
    piano_latent = latent_sample_factor * drums_latent_sample + (1-latent_sample_factor) * drums_latent


    # Use melody NN to convert past piano latent to next piano latent - piano_next_latent: batch_size x K
    piano_next_latent = melody_nn(piano_latent)
    # Add some noise
    random_noise = torch.randn_like(piano_next_latent) * noise_sd
    piano_next_latent = piano_next_latent + random_noise

    # Use conditional NNs to convert piano latent to instrument latent, and add noise - {istr})_next_latent: batch_size x K
    guitar_next_latent = guitar_nn(guitar_latent, piano_next_latent) + torch.randn_like(piano_next_latent) * noise_sd
    bass_next_latent = bass_nn(bass_latent, piano_next_latent) + torch.randn_like(piano_next_latent) * noise_sd
    strings_next_latent = strings_nn(strings_latent, piano_next_latent) + torch.randn_like(piano_next_latent) * noise_sd
    drums_next_latent = drums_nn(drums_latent, piano_next_latent) + torch.randn_like(piano_next_latent) * noise_sd

    # Generate new samples given new latent
    piano_next = piano_vae.generate(piano_next_latent.unsqueeze(0)).view(1, 32, 128)
    guitar_next = guitar_vae.generate(guitar_next_latent.unsqueeze(0)).view(1, 32, 128)
    bass_next = bass_vae.generate(bass_next_latent.unsqueeze(0)).view(1, 32, 128)
    strings_next = strings_vae.generate(strings_next_latent.unsqueeze(0)).view(1, 32, 128)
    drums_next = drums_vae.generate(drums_next_latent.unsqueeze(0)).view(1, 32, 128)

    creation = torch.cat((piano_next, guitar_next, bass_next, strings_next, drums_next), dim = 0)
    creation[creation < threshold] = 0

    if binarize == True:
        creation[creation > 0] = 0.8

    # Quieten the strings
    creation[3, :, :] = creation[3, :, :] * 0.75

    return creation

def load_vaes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Specify dimensionality of VAEs you want (K = 8, 16, 32, 64)
    K = 16

    MODEL_DIR = 'models/'

    # Load VAEs
    model_name = 'VAE_piano_{}'.format(K)
    save_path = MODEL_DIR + model_name
    piano_vae = ConvVAE(K=K).to(device)
    piano_vae.load_state_dict(torch.load(save_path))
    piano_vae.eval()

    model_name = 'VAE_guitar_{}'.format(K)
    save_path = MODEL_DIR + model_name
    guitar_vae = ConvVAE(K=K).to(device)
    guitar_vae.load_state_dict(torch.load(save_path))
    guitar_vae.eval()

    model_name = 'VAE_bass_{}'.format(K)
    save_path = MODEL_DIR + model_name
    bass_vae = ConvVAE(K=K).to(device)
    bass_vae.load_state_dict(torch.load(save_path))
    bass_vae.eval()

    model_name = 'VAE_strings_{}'.format(K)
    save_path = MODEL_DIR + model_name
    strings_vae = ConvVAE(K=K).to(device)
    strings_vae.load_state_dict(torch.load(save_path))
    strings_vae.eval()

    model_name = 'VAE_drums_{}'.format(K)
    save_path = MODEL_DIR + model_name
    drums_vae = ConvVAE(K=K).to(device)
    drums_vae.load_state_dict(torch.load(save_path))
    drums_vae.eval()

    # Load Melody NN
    model_name = 'VAE_NN_piano_{}'.format(K)
    save_path = MODEL_DIR + model_name
    melody_nn = MelodyNN(K = K).to(device)
    melody_nn.load_state_dict(torch.load(save_path))
    melody_nn.eval()

    # Load Conditional NNs
    model_name = 'VAE_NN_guitar_{}'.format(K)
    save_path = MODEL_DIR + model_name
    guitar_nn = ConditionalNN(K = K).to(device)
    guitar_nn.load_state_dict(torch.load(save_path))
    guitar_nn.eval()

    model_name = 'VAE_NN_bass_{}'.format(K)
    save_path = MODEL_DIR + model_name
    bass_nn = ConditionalNN(K = K).to(device)
    bass_nn.load_state_dict(torch.load(save_path))
    guitar_nn.eval()

    model_name = 'VAE_NN_strings_{}'.format(K)
    save_path = MODEL_DIR + model_name
    strings_nn = ConditionalNN(K = K).to(device)
    strings_nn.load_state_dict(torch.load(save_path))
    strings_nn.eval()

    model_name = 'VAE_NN_drums_{}'.format(K)
    save_path = MODEL_DIR + model_name
    drums_nn = ConditionalNN(K = K).to(device)
    drums_nn.load_state_dict(torch.load(save_path))
    drums_nn.eval()
    
    return [piano_vae, guitar_vae, bass_vae, strings_vae, drums_vae], [melody_nn, guitar_nn, bass_nn, strings_nn, drums_nn]

# @title Generate Latent Samples function
def generate_latent_samples(song_loader, vae_models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    piano_vae, guitar_vae, bass_vae, strings_vae, drums_vae = vae_models

    # Get list of the song's latent vectors
    song_latent_list = {'piano': [], 'guitar': [], 'bass': [], 'strings': [], 'drums': []}
    piano_empty_added = False
    guitar_empty_added = False
    bass_empty_added  = False
    strings_empty_added = False
    drums_empty_added = False
    for piano_sequence, guitar_sequence, bass_sequence, strings_sequence, drums_sequence in song_loader:
        if piano_sequence.sum() != 0:
            piano_latent = piano_vae.infer(piano_sequence.to(device))[:, :-1]
            song_latent_list['piano'].append(piano_latent)
        elif piano_empty_added == False:
            piano_latent = piano_vae.infer(piano_sequence.to(device))[:, :-1]
            song_latent_list['piano'].append(piano_latent)
            piano_empty_added = True

        if guitar_sequence.sum() != 0:
            guitar_latent = guitar_vae.infer(guitar_sequence.to(device))[:, :-1]
            song_latent_list['guitar'].append(guitar_latent)
        elif guitar_empty_added == False:
            guitar_latent = guitar_vae.infer(guitar_sequence.to(device))[:, :-1]
            song_latent_list['guitar'].append(guitar_latent)
            guitar_empty_added = True

        if bass_sequence.sum() != 0:
            bass_latent = bass_vae.infer(bass_sequence.to(device))[:, :-1]
            song_latent_list['bass'].append(bass_latent)
        elif bass_empty_added == False:
            bass_latent = bass_vae.infer(bass_sequence.to(device))[:, :-1]
            song_latent_list['bass'].append(bass_latent)
            bass_empty_added = True

        if strings_sequence.sum() != 0:
            strings_latent = strings_vae.infer(strings_sequence.to(device))[:, :-1]
            song_latent_list['strings'].append(strings_latent)
        elif strings_empty_added == False:
            strings_latent = strings_vae.infer(strings_sequence.to(device))[:, :-1]
            song_latent_list['strings'].append(strings_latent)
            strings_empty_added = True

        if drums_sequence.sum() != 0:
            drums_latent = drums_vae.infer(drums_sequence.to(device))[:, :-1]
            song_latent_list['drums'].append(drums_latent)
        elif drums_empty_added == False:
            drums_latent = drums_vae.infer(drums_sequence.to(device))[:, :-1]
            song_latent_list['drums'].append(drums_latent)
            drums_empty_added = True

    return song_latent_list