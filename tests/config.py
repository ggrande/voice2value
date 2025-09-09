import os

config_test_PyannoteVAD = {
            'model_params': {},
            'performance_measurement': False,
            'add_segment_metadata': False,
            'output_dir': 'segmented_audio',
            'huggingface_ACCESS_TOKEN': os.getenv('HUGGINGFACE_ACCESS_TOKEN', ''),
        }