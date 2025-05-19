import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class SequenceDataset(tf.data.Dataset):
    def __new__(cls, embeddings, plddt, target_sequences):
        def _generator():
            assert len(embeddings) == len(plddt) == len(target_sequences)
            for i in range(len(embeddings)):
                yield (embeddings[i].astype(np.float32),
                       plddt[i].astype(np.float32),
                       target_sequences[i].astype(np.float32))

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(1024,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(target_sequences.shape[1],), dtype=tf.float32)
            )
        )

def create_batches_by_length(dataset, batch_size):
    batched_datasets = {}
    lengths = set()
    for _, variable_seq in dataset:
        lengths.add(tf.shape(variable_seq).numpy())

    for length in sorted(list(lengths)):
        filtered_dataset = dataset.filter(
            lambda fixed, variable: tf.shape(variable)[0] == length
        )
        batched_datasets[length] = filtered_dataset.batch(batch_size, drop_remainder=True)

    return batched_datasets


if __name__=="__main__":
    # dummpy data
    total_num_samples = 100
    fixed_data = np.random.rand(total_num_samples, 1024).astype(np.float32)
    variable_data = [np.random.rand(np.random.randint(50, 150)).astype(np.float32) for _ in range(total_num_samples)]
    target_data = np.random.rand(total_num_samples, 128).astype(np.float32)

    # Initialize the dataset
    sequence_dataset = SequenceDataset(fixed_data, variable_data, target_data)

    # Explore the dataset
    # for i, (fixed_sample, variable_sample, target_sample) in enumerate(sequence_dataset.take(5)):
    #     print(f"Sample {i+1}:")
    #     print(f"  Fixed Sequence Shape: {tf.shape(fixed_sample).numpy()}")
    #     print(f"  Variable Sequence Shape: {tf.shape(variable_sample).numpy()}")
    #     print(f"  Target Shape: {tf.shape(target_sample).numpy()}")
    #     print("-" * 20)


    batched_dataset = sequence_dataset.batch(32)
    for fixed_batch, variable_batch, target_batch in batched_dataset.take(1):
        print("Batched Data:")
        print(f"  Fixed Batch Shape: {tf.shape(fixed_batch).numpy()}")
        # Variable batch will be a list of tensors with different shapes
        print(f"  Number of Variable Sequences in Batch: {len(variable_batch)}")
        for i, var_seq in enumerate(variable_batch):
            print(f"    Variable Sequence {i+1} Shape: {tf.shape(var_seq).numpy()}")
        print(f"  Target Batch Shape: {tf.shape(target_batch).numpy()}")
