import yaml
from learn.net import DenseNet
import os
import numpy as np


class GANBuilder:
    def __init__(self, config_path=None):
        config = self._get_config(config_path)
        self.model_config = config['model_config']
        self.dashboard_config = config['dashboard_config']
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.adversarial = self._build_adversarial()

    def _generate_data_real(self, batch_size):
        data_config = self.model_config['target_distribution']
        if data_config['name'] == 'normal':
            return np.random.normal(loc=data_config['mean'], scale=data_config['standard_deviation'],
                                    size=[batch_size, 1])
        else:
            raise LookupError("Invalid data generation type in config")

    def _generate_noise(self, batch_size):
        noise_config = self.model_config['generator_config']
        if noise_config['noise_type'] == 'normal':
            noise = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, noise_config['noise_dim']))
        elif noise_config['noise_type'] == 'uniform':
            noise = np.random.normal(loc=0, scale=1.0, size=(batch_size, noise_config['noise_dim']))
        else:
            LookupError("Invalid noise type in config")
        return noise

    def generate_samples(self, batch_size):
        noise = self._generate_noise(batch_size)
        return np.array([self.generator.activate(n) for n in noise])

    def _train_discriminator_batch(self):
        batch_size = self.model_config['discriminator_config']['batch_size']
        split_size = int(batch_size / 2)
        real_data = self._generate_data_real(split_size)
        fake_data = self.generate_samples(split_size)
        labels = np.concatenate((np.ones(split_size), np.zeros(split_size)))
        shuffling = np.random.permutation(len(labels))

        labels_shuffled = labels[shuffling]
        data_shuffled = np.concatenate((real_data, fake_data))[shuffling]
        print(self.discriminator.update(data_shuffled, labels_shuffled))

    def _train_generator_batch(self):
        batch_size = self.model_config['generator_config']['batch_size']
        noise = self._generate_noise(batch_size)
        labels = np.ones(batch_size)
        print(self.adversarial.update(noise, labels))

    @staticmethod
    def _get_config(config_path):
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def _build_model(params):
        output_model = DenseNet(
            params['input_size'],
            params['fitness'],
            initial_variance=params['initialization_variance']
        )
        for i, layer in enumerate(params['layers']):
            initial_variance = layer['initialization_variance'] \
                if 'initial_variance' in layer else params['initialization_variance']
            output_model.add_layer(
                layer['units'],
                layer['activation'],
                initial_variance
            )
        return output_model

    def _build_generator(self):
        generator_config = self.model_config['generator_config']
        generator_config['input_size'] = generator_config['noise_dim']
        generator_config['fitness'] = 'none'  # The generator doesn't have
        return GANBuilder._build_model(generator_config)

    def _build_discriminator(self):
        discriminator_config = self.model_config['discriminator_config']
        # input size of the discriminator should be the same as the generator's output
        generator_layers = self.model_config['generator_config']['layers']
        discriminator_config['input_size'] = generator_layers[-1]['units']
        discriminator_config['fitness'] = 'cross_entropy_1d'
        return GANBuilder._build_model(discriminator_config)

    def _build_adversarial(self):
        assert self.discriminator is not None, "Discriminator no yet built"
        assert self.generator is not None, "Generator no yet built"
        adversarial_model = DenseNet(self.generator.input_size,
                                     self.discriminator.fitness_function_string,
                                     learning_rate=self.generator.learning_rate,
                                     )
        adversarial_model.frozen = [False for _ in range(len(self.generator.layers))] + [True for _ in range(
            len(self.discriminator.layers))]
        adversarial_model.layers = self.generator.layers + self.discriminator.layers
        return adversarial_model


if __name__ == "__main__":
    gan = GANBuilder('gan_config.yaml')
    for _ in range(100):
        print(gan._train_discriminator_batch())
