
import keras
import tensorflow as tf


def encoder():
    input_embeddings = keras.layers.Input(shape=(None, 1024), name="embeddings")
    # conv_emb_1 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(input_embeddings)
    # conv_emb_2 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_emb_1)
    # conv_emb_3 = keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(conv_emb_2)
    # conv_emb_4 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(conv_emb_3)

    input_plddt = keras.layers.Input(shape=(None, 1), name="plddt")
    # conv_plddt_1 = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(input_plddt)
    # conv_plddt_2 = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(conv_plddt_1)
    # conv_plddt_3 = keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(conv_plddt_2)
    # conv_plddt_4 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu")(conv_plddt_3)
    # merged = keras.layers.concatenate([conv_emb_4, conv_plddt_4], axis=-1)



    x = keras.layers.Conv1D(512, kernel_size=3, padding="same")(input_embeddings)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.2)(x)

    y = keras.layers.Conv1D(8, kernel_size=3, padding="same")(input_plddt)
    y = keras.layers.LeakyReLU()(y)
    y = keras.layers.Dropout(0.2)(y)

    merged = keras.layers.concatenate([x, y], axis=-1)
    merged = keras.layers.Conv1D(512, kernel_size=3, padding="same")(input_plddt)
    merged = keras.layers.LeakyReLU()(merged)
    merged = keras.layers.Dropout(0.2)(merged)    

    return keras.Model(inputs=[input_embeddings, input_plddt], outputs=merged)

def NMR_head():
    combined_features = keras.layers.Input(shape=(None, 512), name="combined_features")
    x = keras.layers.Dense(256)(combined_features)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    output_head_NMR = keras.layers.Dense(1, activation="sigmoid", name="g_scores")(x)
    return keras.Model(inputs=combined_features, outputs=output_head_NMR)

def DisProt_head():
    combined_features = keras.layers.Input(shape=(None, 512), name="combined_features")
    x = keras.layers.Dense(256)(combined_features)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    pooled_output = keras.layers.GlobalAveragePooling1D()(x)
    output_head_DisProt = keras.layers.Dense(1, activation="sigmoid", name="disorder_content")(pooled_output)
    return keras.Model(inputs=combined_features, outputs=output_head_DisProt)

def SoftDis_head():
    combined_features = keras.layers.Input(shape=(None, 512), name="combined_features")
    x = keras.layers.Dense(256, activation="relu")(combined_features)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    output_head_SoftDis = keras.layers.Dense(1, activation="sigmoid", name="soft_disorder_frequency")(x)
    return keras.Model(inputs=combined_features, outputs=output_head_SoftDis)


def masked_loss(y_true, y_pred, mask):
    mask = tf.cast(mask, y_pred.dtype)
    se = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(se) / tf.reduce_sum(mask)


class DeepEntropy(keras.Model):
    def __init__(self, encoder_model = None, nmr_head = None, disProt_head = None, softDis_head = None):
        super().__init__()
        self.NMR_head_loss_tracker = keras.metrics.Mean(name="NMR_head_loss")
        self.DisProt_head_loss_tracker = keras.metrics.Mean(name="DisProt_head_loss")
        self.SoftDis_head_loss_tracker = keras.metrics.Mean(name="SoftDis_head_loss")

        if nmr_head:
            self.nmr_head_model = nmr_head
        else:
            self.nmr_head_model = NMR_head()

        if disProt_head:
            self.DisProt_head_model = disProt_head
        else:
            self.DisProt_head_model = DisProt_head()

        if softDis_head:
            self.softdis_head_model = softDis_head
        else:
            self.softdis_head_model = SoftDis_head()

        if encoder_model:
            self.encoder = encoder_model
        else:
            self.encoder = encoder()

        self.target_flag_ = None

        self.NMR_head_loss = []
        self.DisProt_head_loss = []
        self.SoftDis_head_loss = []

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


    def build_init(self, embedding_shape, plddt_shape, encoded_data_dim=512):
        dummy_embeddings = tf.zeros(embedding_shape)
        dummy_plddt = tf.zeros(plddt_shape)

        _ = self.encoder([dummy_embeddings, dummy_plddt])
        _ = self.nmr_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], encoded_data_dim)))
        _ = self.DisProt_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], encoded_data_dim)))
        _ = self.softdis_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], encoded_data_dim)))

        print("All the heads and optimizers are initialized. Good luck with the training loop!")


    # @tf.function
    def train_step(self, data):
        ((input_embeddings, input_plddt, target, mask_tf),_,_) = data
        # ((input_embeddings, input_plddt, target, mask_tf),) = data

        if self.target_flag_=="g_scores":
            with tf.GradientTape() as tape:
                combined_features = self.encoder([input_embeddings, input_plddt])
                predicted_g_scores = self.nmr_head_model(combined_features)
                # g_loss = self.NMR_head_loss_fn(target, predicted_g_scores, mask=mask_tf)
                g_loss = self.NMR_head_loss_fn(target, predicted_g_scores)

            trainable_variables = self.encoder.trainable_variables + self.nmr_head_model.trainable_variables
            gradients = tape.gradient(g_loss, trainable_variables)
            self.NMR_optimizer.apply_gradients(zip(gradients, trainable_variables))
            self.NMR_head_loss_tracker.update_state(g_loss)
            self.NMR_head_loss.append(self.NMR_head_loss_tracker.result())
            loss_metrics =  {
                "NMR_head_loss": self.NMR_head_loss_tracker.result(),
                "DisProt_head_loss": tf.constant(float("nan")),
                "SoftDis_head_loss": tf.constant(float("nan"))
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
                "NMR_head_loss": tf.constant(float("nan")),
                "DisProt_head_loss": self.DisProt_head_loss_tracker.result(),
                "SoftDis_head_loss": tf.constant(float("nan"))
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
                "NMR_head_loss": tf.constant(float("nan")),
                "DisProt_head_loss": tf.constant(float("nan")),
                "SoftDis_head_loss": self.SoftDis_head_loss_tracker.result()
            }

        return loss_metrics

    def predict(self, data):
        (input_embeddings, input_plddt) = data
        combined_features = self.encoder([input_embeddings, input_plddt])

        if self.target_flag_ == "g_scores":
            return self.nmr_head_model(combined_features)
        elif self.target_flag_ == "disprot_disorder":
            return self.DisProt_head_model(combined_features)
        else:
            return self.softdis_head_model(combined_features)
        

if __name__ == '__main__':
    entropy = DeepEntropy()
