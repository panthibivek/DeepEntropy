
import keras
import tensorflow as tf


def encoder():
    input_embeddings = keras.layers.Input(shape=(None, 1024), ragged=True, name="embeddings")
    conv_emb_1 = keras.layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(input_embeddings)
    conv_emb_2 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(conv_emb_1)
    conv_emb_3 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_emb_2)
    conv_emb_4 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_emb_3)


    input_plddt = keras.layers.Input(shape=(None, 1), ragged=True, name="plddt")
    conv_plddt_1 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(input_plddt)
    conv_plddt_2 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_plddt_1)
    conv_plddt_3 = keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(conv_plddt_2)
    conv_plddt_4 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(conv_plddt_3)

    merged = keras.layers.concatenate([conv_emb_4, conv_plddt_4], axis=-1)
    return keras.Model(inputs=[input_embeddings, input_plddt], outputs=merged)


def NMR_head():
    combined_features = keras.layers.Input(shape=(None, 40), ragged=True, name="combined_features")
    head_NMR_1 = keras.layers.Dense(64, activation="relu")(combined_features)
    head_NMR_2 = keras.layers.Dense(32, activation="relu")(head_NMR_1)
    output_head_NMR = keras.layers.Dense(1, activation="relu", name="g_scores")(head_NMR_2)
    return keras.Model(inputs=combined_features, outputs=output_head_NMR)


def DisProt_head():
    combined_features = keras.layers.Input(shape=(None, 40), ragged=True, name="combined_features")
    head_DisProt_1 = keras.layers.Dense(64, activation="relu")(combined_features)
    head_DisProt_2 = keras.layers.Dense(32, activation="relu")(head_DisProt_1)
    pooled_output = keras.layers.GlobalAveragePooling1D()(head_DisProt_2)
    output_head_DisProt = keras.layers.Dense(1, activation="sigmoid", name="disorder_content")(pooled_output)
    return keras.Model(inputs=combined_features, outputs=output_head_DisProt)


def SoftDis_head():
    combined_features = keras.layers.Input(shape=(None, 40), ragged=True, name="combined_features")
    head_SoftDis_1 = keras.layers.Dense(64, activation="relu")(combined_features)
    head_SoftDis_2 = keras.layers.Dense(32, activation="relu")(head_SoftDis_1)
    output_head_SoftDis = keras.layers.Dense(1, activation="sigmoid", name="soft_disorder_frequency")(head_SoftDis_2)
    return keras.Model(inputs=combined_features, outputs=output_head_SoftDis)



class DeepEntropy(keras.Model):
    def __init__(self):
        super().__init__()

        self.NMR_head_loss_tracker = keras.metrics.Mean(name="NMR_head_loss")
        self.DisProt_head_loss_tracker = keras.metrics.Mean(name="DisProt_head_loss")
        self.SoftDis_head_loss_tracker = keras.metrics.Mean(name="SoftDis_head_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.nmr_head_model = NMR_head()
        self.DisProt_head_model = DisProt_head()
        self.softdis_head_model = SoftDis_head()
        self.encoder = encoder()

        self.NMR_head_loss = []
        self.DisProt_head_loss = []
        self.SoftDis_head_loss = []
        self.total_loss = []


    @property
    def metrics(self):
        return [self.NMR_head_loss_tracker, self.DisProt_head_loss_tracker, self.SoftDis_head_loss_tracker]


    def compile(self, optimizer, NMR_head_loss_fn, DisProt_head_loss_fn, SoftDis_head_loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.NMR_head_loss_fn = NMR_head_loss_fn
        self.DisProt_head_loss_fn = DisProt_head_loss_fn
        self.SoftDis_head_loss_fn = SoftDis_head_loss_fn


    def train_step(self, input_embeddings, input_plddt, g_scores_target, disorder_target, soft_disorder_target):
        with tf.GradientTape() as tape:
            combined_features = self.encoder([input_embeddings, input_plddt])
            predicted_g_scores = self.nmr_head_model(tf.expand_dims(combined_features, axis=1))
            predicted_disorder = self.DisProt_head_model(tf.expand_dims(combined_features, axis=1))
            predicted_soft_disorder = self.softdis_head_model(tf.expand_dims(combined_features, axis=1))

            g_loss = self.NMR_head_loss_fn(g_scores_target, predicted_g_scores)
            disorder_loss = self.DisProt_head_loss_fn(disorder_target, predicted_disorder)
            soft_disorder_loss = self.SoftDis_head_loss_fn(soft_disorder_target, predicted_soft_disorder)
            total_loss = g_loss + disorder_loss + soft_disorder_loss

        trainable_variables = self.encoder.trainable_variables + \
                            self.nmr_head_model.trainable_variables + \
                            self.DisProt_head_model.trainable_variables + \
                            self.softdis_head_model.trainable_variables

        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # loss tracking
        self.NMR_head_loss_tracker.update_state(g_loss)
        self.DisProt_head_loss_tracker.update_state(disorder_loss)
        self.SoftDis_head_loss_tracker.update_state(soft_disorder_loss)
        self.total_loss_tracker.update_state(total_loss)

        self.NMR_head_loss.append(self.NMR_head_loss_tracker.result())
        self.DisProt_head_loss.append(self.DisProt_head_loss_tracker.result())
        self.SoftDis_head_loss.append(self.SoftDis_head_loss_tracker.result())
        self.total_loss.append(self.total_loss_tracker.result())

        loss_metrics =  {
            "NMR_head_loss": self.NMR_head_loss_tracker.result(),
            "DisProt_head_loss": self.DisProt_head_loss_tracker.result(),
            "SoftDis_head_loss": self.SoftDis_head_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }
        return loss_metrics



if __name__ == '__main__':
    entropy = DeepEntropy()
