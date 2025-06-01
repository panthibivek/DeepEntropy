
import keras
import tensorflow as tf


def encoder():
    input_embeddings = keras.layers.Input(shape=(None, 1024), name="embeddings")
    conv_emb_1 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(input_embeddings)
    conv_emb_2 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_emb_1)
    conv_emb_3 = keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(conv_emb_2)
    conv_emb_4 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(conv_emb_3)


    input_plddt = keras.layers.Input(shape=(None, 1), name="plddt")
    conv_plddt_1 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(input_plddt)
    conv_plddt_2 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_plddt_1)
    conv_plddt_3 = keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(conv_plddt_2)
    conv_plddt_4 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(conv_plddt_3)

    merged = keras.layers.concatenate([conv_emb_4, conv_plddt_4], axis=-1)
    return keras.Model(inputs=[input_embeddings, input_plddt], outputs=merged)


def NMR_head():
    combined_features = keras.layers.Input(shape=(None, 16), name="combined_features")
    head_NMR_1 = keras.layers.Dense(64, activation="relu")(combined_features)
    head_NMR_2 = keras.layers.Dense(32, activation="relu")(head_NMR_1)
    output_head_NMR = keras.layers.Dense(1, activation="relu", name="g_scores")(head_NMR_2)
    return keras.Model(inputs=combined_features, outputs=output_head_NMR)


def DisProt_head():
    combined_features = keras.layers.Input(shape=(None, 16), name="combined_features")
    head_DisProt_1 = keras.layers.Dense(64, activation="relu")(combined_features)
    head_DisProt_2 = keras.layers.Dense(32, activation="relu")(head_DisProt_1)
    pooled_output = keras.layers.GlobalAveragePooling1D()(head_DisProt_2)
    output_head_DisProt = keras.layers.Dense(1, activation="sigmoid", name="disorder_content")(pooled_output)
    return keras.Model(inputs=combined_features, outputs=output_head_DisProt)


def SoftDis_head():
    combined_features = keras.layers.Input(shape=(None, 16), name="combined_features")
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

        self.nmr_head_model = NMR_head()
        self.DisProt_head_model = DisProt_head()
        self.softdis_head_model = SoftDis_head()
        self.encoder = encoder()
        self.target_flag_ = None

        self.NMR_head_loss = []
        self.DisProt_head_loss = []
        self.SoftDis_head_loss = []
        self.total_loss = []


    # @property
    # def metrics(self):
    #     return [self.NMR_head_loss_tracker, self.DisProt_head_loss_tracker, self.SoftDis_head_loss_tracker]


    def compile(self, NMR_optimizer, DisProt_optimizer, SoftDis_optimizer, NMR_head_loss_fn, DisProt_head_loss_fn, SoftDis_head_loss_fn):
        super().compile()
        self.NMR_optimizer = NMR_optimizer
        self.DisProt_optimizer = DisProt_optimizer
        self.SoftDis_optimizer = SoftDis_optimizer

        self.NMR_head_loss_fn = NMR_head_loss_fn
        self.DisProt_head_loss_fn = DisProt_head_loss_fn
        self.SoftDis_head_loss_fn = SoftDis_head_loss_fn


    def train_step(self, data):
        ((input_embeddings, input_plddt, target),) = data
        # (input_data, target) = data
        # (input_embeddings, input_plddt) = input_data

        if self.target_flag_=="g_scores":
            with tf.GradientTape() as tape:
                combined_features = self.encoder([input_embeddings, input_plddt])
                predicted_g_scores = self.nmr_head_model(combined_features)
                g_loss = self.NMR_head_loss_fn(target, predicted_g_scores)

            trainable_variables = self.encoder.trainable_variables + self.nmr_head_model.trainable_variables
            gradients = tape.gradient(g_loss, trainable_variables)
            self.NMR_optimizer.apply_gradients(zip(gradients, trainable_variables))
            self.NMR_head_loss_tracker.update_state(g_loss)
            self.NMR_head_loss.append(self.NMR_head_loss_tracker.result())
            loss_metrics =  {
                "loss": self.NMR_head_loss_tracker.result()
            }
            
        elif self.target_flag_=="disprot_disorder":
            with tf.GradientTape() as tape:
                combined_features = self.encoder([input_embeddings, input_plddt])
                predicted_disorder = self.DisProt_head_model(combined_features)
                disorder_loss = self.DisProt_head_loss_fn(target, predicted_disorder)

            trainable_variables = self.encoder.trainable_variables + self.DisProt_head_model.trainable_variables
            gradients = tape.gradient(disorder_loss, trainable_variables)
            self.DisProt_optimizer.apply_gradients(zip(gradients, trainable_variables))
            self.DisProt_head_loss_tracker.update_state(disorder_loss)
            self.DisProt_head_loss.append(self.DisProt_head_loss_tracker.result())
            loss_metrics =  {
                "loss": self.DisProt_head_loss_tracker.result()
            }

        else:
            with tf.GradientTape() as tape:
                combined_features = self.encoder([input_embeddings, input_plddt])
                predicted_soft_disorder = self.softdis_head_model(combined_features)
                soft_disorder_loss = self.SoftDis_head_loss_fn(target, predicted_soft_disorder)

            trainable_variables = self.encoder.trainable_variables + self.softdis_head_model.trainable_variables
            gradients = tape.gradient(soft_disorder_loss, trainable_variables)
            self.SoftDis_optimizer.apply_gradients(zip(gradients, trainable_variables))
            self.SoftDis_head_loss_tracker.update_state(soft_disorder_loss)
            self.SoftDis_head_loss.append(self.SoftDis_head_loss_tracker.result())
            loss_metrics =  {
                "loss": self.SoftDis_head_loss_tracker.result()
            }

        return loss_metrics



if __name__ == '__main__':
    entropy = DeepEntropy()
