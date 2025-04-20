import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR     = "/home/josh/Desktop/Imaging"
SEG_LABEL    = "CT1"  # Use "CT1" as tumor segmentation label
IMG_SIZE     = (128, 128)
SEQ_LEN      = 3
LATENT_DIM   = 128
BATCH_SIZE   = 4
EPOCHS       = 75
MODEL_SAVE   = "vae_convlstm_model.keras"

# ─── DATA LOADING ────────────────────────────────────────────────────────────────
def load_patient_slices(patient_dir):
    """
    Load binary tumor mask slices for a patient by checking:
    1) directly in each week-XXX folder,
    2) then in HD-GLIO-AUTO-segmentation/native subfolder.
    """
    slices = []
    weeks = sorted(
        d for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d)) and d.startswith("week-")
    )
    print(f"Checking {len(weeks)} weeks in {patient_dir}...")
    for w in weeks:
        week_dir = os.path.join(patient_dir, w)
        # First, check directly in week folder
        seg_root = week_dir
        # Then fallback to segmentation subfolder
        seg_fallback = os.path.join(week_dir, "HD-GLIO-AUTO-segmentation", "native")

        seg_files = []
        # Look for CT1 files in week folder
        if os.path.isdir(seg_root):
            seg_files += [f for f in os.listdir(seg_root)
                          if "CT1" in f and not f.startswith("T1") and f.endswith(".nii.gz")]
        # If none found, look in fallback directory
        if not seg_files and os.path.isdir(seg_fallback):
            seg_files += [f for f in os.listdir(seg_fallback)
                          if "CT1" in f and not f.startswith("T1") and f.endswith(".nii.gz")]
        if not seg_files:
            print(f"[SKIP] No segmentation file with 'CT1' in {week_dir} or fallback {seg_fallback}")
            continue
        # Use first matching file
        file_name = seg_files[0]
        source_dir = seg_root if file_name in os.listdir(seg_root) else seg_fallback
        fpath = os.path.join(source_dir, file_name)
        if not os.path.exists(fpath):
            print(f"[SKIP] Missing: {fpath}")
            continue
        # Load and process
        try:
            vol = nib.load(fpath).get_fdata()
            mid = vol.shape[2] // 2
            img = vol[..., mid]
            img = (img > 0).astype(np.float32)
            img = tf.image.resize(img[..., None], IMG_SIZE).numpy()
            slices.append(img)
        except Exception as e:
            print(f"[ERROR] Failed loading {fpath}: {e}")
    return slices

# ─── SEQUENCE BUILDING & DATASET LOADING ──────────────────────────────────────────
def build_sequences(slices):
    X, Y = [], []
    for i in range(len(slices) - SEQ_LEN):
        X.append(np.stack(slices[i:i + SEQ_LEN], axis=0))
        Y.append(slices[i + SEQ_LEN])
    return np.array(X), np.array(Y)

def load_dataset():
    Xs, Ys = [], []
    patients_checked = 0
    patients_used = 0

    for p in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, p)
        if not os.path.isdir(pdir) or not p.startswith("Patient-"):
            continue

        patients_checked += 1
        s = load_patient_slices(pdir)
        if len(s) < SEQ_LEN + 1:
            print(f"[SKIP] {p} only has {len(s)} usable slices")
            continue

        x, y = build_sequences(s)
        if len(x) > 0 and len(y) > 0:
            Xs.append(x)
            Ys.append(y)
            patients_used += 1

    if not Xs or not Ys:
        raise ValueError(f"No valid data found. Checked {patients_checked} patients, used {patients_used}.")

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    return (X[idx[:split]], Y[idx[:split]]), (X[idx[split:]], Y[idx[split:]])

# ─── VAE CONVLSTM MODEL ──────────────────────────────────────────────────────────
class VAE_ConvLSTM(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, activation='relu'),
            layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=False, activation='relu'),
            layers.Flatten(),
        ])
        self.z_mean = layers.Dense(LATENT_DIM, name='z_mean')
        self.z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')
        self.decoder_fc = layers.Dense(32 * 32 * 32, activation='relu')
        self.decoder_reshape = layers.Reshape((32, 32, 32))
        self.decoder_up1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')
        self.decoder_up2 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')
        self.decoder_out = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def sample(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        x = self.encoder(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sample(z_mean, z_log_var)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)

        x = self.decoder_fc(z)
        x = self.decoder_reshape(x)
        x = self.decoder_up1(x)
        x = self.decoder_up2(x)
        return self.decoder_out(x)

# ─── LOSS ───────────────────────────────────────────────────────────────────────
# ─── TOTAL VARIATION LOSS ────────────────────────────────────────────────────────
def total_variation_loss(y_pred):
    return tf.reduce_mean(tf.image.total_variation(y_pred))

# ─── COMBINED HYBRID LOSS (MSE + SSIM + TV) ──────────────────────────────────────
def combined_hybrid_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    tv_loss = total_variation_loss(y_pred)

    # Adjust these weights as needed
    alpha, beta, gamma = 1.0, 0.1, 0.0

    return alpha * mse_loss + beta * ssim_loss + gamma * tv_loss
# ─── TRAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (train_X, train_y), (val_X, val_y) = load_dataset()
    print("Train:", train_X.shape, train_y.shape)
    print("Val:  ", val_X.shape, val_y.shape)

    model = VAE_ConvLSTM()
    model.compile(
        optimizer="adam",
        loss=combined_hybrid_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    model.summary()
    model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    model.save(MODEL_SAVE)
    print("Model saved to", MODEL_SAVE)

    preds = model.predict(val_X)
    mse = np.mean((preds - val_y) ** 2)
    ssim_vals = tf.image.ssim(
        tf.convert_to_tensor(val_y, tf.float32),
        tf.convert_to_tensor(preds, tf.float32),
        max_val=1.0
    )
    print("Validation MSE:", mse)
    print("Validation SSIM:", float(tf.reduce_mean(ssim_vals)))
