from holodecml.models import ParticleEncoder, ParticleAttentionNet, generate_gaussian_particles
import numpy as np


def test_particlencoder():
    net = ParticleEncoder()
    assert net.hidden_neurons == 10


def test_particleeattentionnet():
    net = ParticleAttentionNet()
    num_images = 1000
    num_particles = 5
    image_size_pixels = 100
    filter_size = 3
    noise_sd = 0.2
    particle_pos, holo = generate_gaussian_particles(num_images=num_images, num_particles=num_particles, image_size_pixels=image_size_pixels,
                                gaussian_sd=filter_size)
    particle_pos_noisy = particle_pos * (1 + np.random.normal(0, noise_sd, particle_pos.shape))
    net.compile(optimizer="adam", loss="mse")
    net.fit([particle_pos_noisy, holo], particle_pos, epochs=10, batch_size=64, verbose=1)
    assert net.hidden_neurons == 100
    return
