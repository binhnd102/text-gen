import os
import tensorflow as tf

# checkpoint_dir = 'drive/MyDrive/chotot_training/char_generate_phukien/'
# Name of the checkpoint files
def get_callbacks(checkpoint_dir):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=False, filepath=checkpoint_prefix, save_weights_only=True
    )

    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=1,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    return [checkpoint_callback]
