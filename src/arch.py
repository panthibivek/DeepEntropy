
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
    x = keras.layers.Dense(64)(combined_features)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.2)(x)

    output_head_NMR = keras.layers.Dense(1, activation="sigmoid", name="g_scores")(x)
    return keras.Model(inputs=combined_features, outputs=output_head_NMR)

def DisProt_head():
    combined_features = keras.layers.Input(shape=(None, 16), name="combined_features")
    x = keras.layers.Dense(64)(combined_features)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    pooled_output = keras.layers.GlobalAveragePooling1D()(x)
    output_head_DisProt = keras.layers.Dense(1, activation="sigmoid", name="disorder_content")(pooled_output)
    return keras.Model(inputs=combined_features, outputs=output_head_DisProt)

def SoftDis_head():
    combined_features = keras.layers.Input(shape=(None, 16), name="combined_features")
    x = keras.layers.Dense(64, activation="relu")(combined_features)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    output_head_SoftDis = keras.layers.Dense(1, activation="sigmoid", name="soft_disorder_frequency")(x)
    return keras.Model(inputs=combined_features, outputs=output_head_SoftDis)


def masked_loss(y_true, y_pred, mask):
    mask = tf.cast(mask, y_pred.dtype)
    se = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(se) / tf.reduce_sum(mask)


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


    def build_init(self, embedding_shape, plddt_shape):
        dummy_embeddings = tf.zeros(embedding_shape)
        dummy_plddt = tf.zeros(plddt_shape)

        _ = self.encoder([dummy_embeddings, dummy_plddt])
        _ = self.nmr_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], 16)))
        _ = self.DisProt_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], 16)))
        _ = self.softdis_head_model(tf.zeros((embedding_shape[0], embedding_shape[1], 16)))

        print("All the heads and optimizers are initialized. Good luck with the training loop!")


    # def build_init(self, embedding_shape, plddt_shape):
    #     input_embeddings = tf.zeros(embedding_shape, dtype=tf.float32)
    #     input_plddt = tf.zeros(plddt_shape, dtype=tf.float32)
    #     self.encoder.build([embedding_shape, plddt_shape])
    #     self.nmr_head_model.build((None, 16))
    #     self.DisProt_head_model.build((None, 16))
    #     self.softdis_head_model.build((None, 16))

    #     # NMR
    #     with tf.GradientTape() as tape:
    #         combined = self.encoder([input_embeddings, input_plddt])
    #         predicted_g_scores = self.nmr_head_model(combined)
    #         dummy_target_g = tf.zeros_like(input_plddt)
    #         dummy_mask = tf.ones_like(dummy_target_g)
    #         g_loss = self.NMR_head_loss_fn(dummy_target_g, predicted_g_scores, mask=dummy_mask)

    #     g_vars = self.encoder.trainable_variables + self.nmr_head_model.trainable_variables
    #     g_grads = tape.gradient(g_loss, g_vars)
    #     g_grads = [tf.stop_gradient(g) if g is not None else None for g in g_grads]
    #     self.NMR_optimizer.apply_gradients(zip(g_grads, g_vars))

    #     # DisProt
    #     with tf.GradientTape() as tape:
    #         combined = self.encoder([input_embeddings, input_plddt])
    #         predicted_disorder = self.DisProt_head_model(combined)
    #         dummy_target_d = tf.zeros((1, 1), dtype=tf.float32)
    #         d_loss = self.DisProt_head_loss_fn(dummy_target_d, predicted_disorder)

    #     d_vars = self.encoder.trainable_variables + self.DisProt_head_model.trainable_variables
    #     d_grads = tape.gradient(d_loss, d_vars)
    #     d_grads = [tf.stop_gradient(g) if g is not None else None for g in d_grads]
    #     self.DisProt_optimizer.apply_gradients(zip(d_grads, d_vars))

    #     # SoftDis
    #     with tf.GradientTape() as tape:
    #         combined = self.encoder([input_embeddings, input_plddt])
    #         predicted_soft_disorder = self.softdis_head_model(combined)
    #         dummy_target_s = tf.zeros_like(input_plddt)
    #         s_loss = self.SoftDis_head_loss_fn(dummy_target_s, predicted_soft_disorder)

    #     s_vars = self.encoder.trainable_variables + self.softdis_head_model.trainable_variables
    #     s_grads = tape.gradient(s_loss, s_vars)
    #     s_grads = [tf.stop_gradient(g) if g is not None else None for g in s_grads]
    #     self.SoftDis_optimizer.apply_gradients(zip(s_grads, s_vars))

    #     print("All the heads and optimizers are initialized. Good luck with the training loop!")


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
