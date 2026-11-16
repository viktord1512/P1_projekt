
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find og indlæs datasættet
MAPPE = Path(__file__).resolve().parent
heart = pd.read_csv(MAPPE / "heart.csv")

# Numeriske indikatorer
numerical_features = {
    "Age": "Alder (år)",
    "RestingBP": "Blodtryk i hvile (mmHg)",
    "Cholesterol": "Kolesterol (mg/dL)"
}

# Kategoriske indikatorer
categorical_features = {
    "Sex": "Køn",
    "ChestPainType": "Type af brystsmerter",
    "FastingBS": "Fastende blodsukker",
    "ExerciseAngina": "Belastningsudløste brystsmerter"
}

# Opret figur med 2 rækker og 4 kolonner
fig, axes = plt.subplots(
    2, 4,
    figsize=(18, 9),
    layout="constrained"
)

axes = axes.flatten()

# Histogrammer for numeriske indikatorer
for i, (column, title) in enumerate(numerical_features.items()):

    axes[i].hist(
        heart[column],
        bins=20,
        edgecolor="black"
    )

    axes[i].set_title(title, fontsize=12, pad=12)
    axes[i].set_ylabel("Antal patienter")
    axes[i].grid(axis="y", alpha=0.2)

# Søjlediagrammer for kategoriske indikatorer
for i, (column, title) in enumerate(
    categorical_features.items(), start=3
):

    counts = heart[column].value_counts().sort_index()

    axes[i].bar(
        counts.index.astype(str),
        counts.values,
        edgecolor="black"
    )

    axes[i].set_title(title, fontsize=12, pad=12)
    axes[i].set_ylabel("Antal patienter")
    axes[i].grid(axis="y", alpha=0.2)

# Fjern den sidste tomme plads
axes[7].axis("off")

# Hovedoverskrift
fig.suptitle(
    "Fordeling af udvalgte kliniske indikatorer",
    fontsize=18,
    fontweight="bold"
)

# Gem figuren til Overleaf
fig.savefig(
    MAPPE / "histogrammer_udvalgte_opdateret.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
