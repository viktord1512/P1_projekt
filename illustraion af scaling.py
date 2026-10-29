import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, Normalizer
)

plt.style.use("ggplot")

heart = pd.read_csv("heart.csv")

categorical_cols = ["Sex", "ChestPainType", "ExerciseAngina"]
numeric_cols = ["Age", "RestingBP", "Cholesterol"]

# 0 betyder "mangler" i disse kolonner -> gør dem til NaN
heart[["RestingBP", "Cholesterol"]] = heart[["RestingBP", "Cholesterol"]].replace(0, np.nan)

# One-hot encoding (bruges til modellen, ikke til plottet)
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart[categorical_cols]).toarray()

# Numeriske features
X_numerical = heart[numeric_cols]

# Skalerere der skal sammenlignes
scalers = {
    "Standard Scaler": StandardScaler(),
    "Robust Scaler": RobustScaler(),
    "MinMax Scaler": MinMaxScaler(),
    "Normalizer": Normalizer(),
}

scaled_data = {}
for name, scaler in scalers.items():
    if name == "Normalizer":
        # Normalizer kan ikke håndtere NaN -> fjern rækker med manglende værdier
        X_input = X_numerical.dropna()
    else:
        X_input = X_numerical  # NaN ignoreres ved fit og bevares
    scaled_data[name] = pd.DataFrame(
        scaler.fit_transform(X_input),
        columns=numeric_cols,
    )

# Plot før og efter
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()


axes[0].set_title("Before Scaling")
for col in numeric_cols:
    sns.kdeplot(X_numerical[col], ax=axes[0], label=col)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].legend()

for ax, (name, df_scaled) in zip(axes[1:], scaled_data.items()):
    ax.set_title(f"After {name}")
    for col in numeric_cols:
        sns.kdeplot(df_scaled[col], ax=ax, label=col)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.legend()

# Skjul det sidste tomme felt (2x3 = 6 felter, 5 bruges)
axes[5].axis("off")

plt.tight_layout()
plt.show()