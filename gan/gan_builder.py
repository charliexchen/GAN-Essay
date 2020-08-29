import yaml
from GANEssay.learn.net import DenseNet
import os


class GanBuilder:
    def __init__(self, config_path=None):
        self.config = _get_config(config_path)

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

    @staticmethod
    def _build_generator(model_config):
        generator_config = model_config['generator_config']
        generator_config['input_size'] = generator_config['noise_dim']
        generator_config['fitness'] = 'none'  # The generator doesn't have
        return GanBuilder._build_model(generator_config)

    @staticmethod
    def _build_discriminator(model_config):
        discriminator_config = model_config['discriminator_config']
        # input size of the discriminator should be the same as the generator's output
        generator_layers = model_config['generator_config']['layers']
        discriminator_config['input_size'] = generator_layers[-1]['units']
        discriminator_config['fitness'] = 'cross_entropy'
        return GanBuilder._build_model(discriminator_config)

    @staticmethod
    def _build_adversarial(discriminator_model, generator_model):
        adversarial_model = DenseNet(generator_model.input_size,
                                     discriminator_model.fitness_function_string,
                                     learning_rate=generator_model.learning_rate,
                                     )
        adversarial_model.frozen = [False for _ in range(len(generator_model.layers))] + [True for _ in range(
            len(discriminator_model.layers))]
        adversarial_model.layers = generator_model.layers + discriminator_model.layers
        return adversarial_model


with open('gan_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    model_config = config['model_config']

    disc_model = GanBuilder._build_discriminator(model_config)
    print(disc_model)
    generator_model = GanBuilder._build_generator(model_config)
    print(generator_model)
    print(GanBuilder._build_adversarial(disc_model, generator_model))
